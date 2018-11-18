import os, sys, glob
import numpy as np
import pydicom as dicom
from skimage.draw import polygon
import nibabel as nib

def read_structure(structure):
    """
    INPUT:
        structure: RTSS structure file
    OUTPUT:
        contours: a list of structures, where each structure is a dict with structure name, color, number & coordinates
    """
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        # contour['number'] = structure.ROIContourSequence[i].RefdROINumber
        contour['number'] = structure.StructureSetROISequence[i].ROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        # assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours

def get_mask(contours, slices):
    """
    INPUT:
        coutours: output from read_structure; a list of structures
        slices: a list of dicom slices of CT scan corresponding to contours
    OUTPUT:
        label: a mask of the original CT scan where the values correspond to structure number
        colors: a list of colors corresponding
    """
    z = [s.ImagePositionPatient[2] for s in slices]
    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]

    label = np.zeros_like(image, dtype=np.uint8)
    for con in contours:
        num = int(con['number'])
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            zNew = [round(elem,1) for elem in z ]
            try:
                z_index = z.index(np.around(nodes[0, 2],1))
            except ValueError:
                z_index = zNew.index(np.around(nodes[0, 2],1))
            # z_index = z.index(np.around(nodes[0, 2],1))
            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            label[rr, cc, z_index] = num
    colors = tuple(np.array([con['color'] for con in contours]) / 255.0)
    return label, colors

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

## Tip: To save storage space, don't do normalization and zero centering beforehand,
## but do this online (during training, just after loading). If you don't do this yet,
## your image are int16's, which are smaller than float32s and easier to compress as well.

if __name__ == "__main__":
    dataset_directory = './data/Head-Neck-PET-CT'
    output_directory = os.path.join(dataset_directory+"_preprocessed")
    patient_list = [os.path.join(dataset_directory, name) for name in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, name))]

    for patient in patient_list:
        for subdir, dirs, files in os.walk(patient):
            dcms = glob.glob(os.path.join(subdir, "*.dcm"))
            if len(dcms) == 1:
                structure = dicom.read_file(os.path.join(subdir, files[0]))
                contours = read_structure(structure)
            elif len(dcms) > 1:
                slices = [dicom.read_file(dcm) for dcm in dcms]
                slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
                image = np.stack([s.pixel_array for s in slices], axis=-1)

        label, colors = get_mask(contours, slices) #colors can be used to visualize using pyplot

        #convert to hounsfeld unit
        image = get_pixels_hu(slices)

        # to check output in viewer, export image and label as nii
        # label_nii = nib.Nifti1Image(label, np.eye(4))
        # image_nii = nib.Nifti1Image(image, np.eye(4))
        # label_nii.to_filename('label.nii')
        # image_nii.to_filename('image.nii')

        np.save(os.path.join(output_directory, os.path.basename(patient)+"_CTVOL.npy"), image)
        np.save(os.path.join(output_directory, os.path.basename(patient)+"_label.npy"), label)
