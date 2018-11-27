import os, sys, glob
import numpy as np
import pydicom as dicom
from skimage.draw import polygon
import nibabel as nib
import scipy.ndimage
import shutil

"""
Preprocess script for DICOM CT scans and RTSS structure files
Specifically for Radiation Oncology CT simulation scans
"""

def read_structure(structure):
    """
    INPUT:
        structure: RTSS structure file
    OUTPUT:
        contours: a list of structures, where each structure is a dict with structure name, color, number & coordinates
    """
    desired_structures = ['peau', 'body', 'external', 'externe']
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        # print(structure.StructureSetROISequence[i].ROIName)
        if structure.StructureSetROISequence[i].ROIName.lower() not in desired_structures:
            continue
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        # contour['number'] = structure.ROIContourSequence[i].RefdROINumber
        contour['number'] = structure.StructureSetROISequence[i].ROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        # assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)

    if len(contours) == 0:
        print('No desired structures found.')
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

    image = np.stack([s.pixel_array for s in slices], axis=-1)
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
            zNew = [round(elem,0) for elem in z ]
            try:
                z_index = z.index(round(nodes[0, 2], 1))
            except ValueError:
                try:
                    z_index = zNew.index(round(nodes[0, 2], 0))
                except ValueError:
                    z_index = (np.abs(z - nodes[0, 2])).argmin()
            # z_index = z.index(np.around(nodes[0, 2],1))
            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            # label[rr, cc, z_index] = num
            label[rr, cc, z_index] = int(1)
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
            image[:, :, slice_number] = slope * image[slice_number].astype(np.float64)
            image[:, :, slice_number] = image[slice_number].astype(np.int16)

        image[:, :, slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, slices, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array(list(slices[0].PixelSpacing) + [slices[0].SliceThickness], dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def resample_to_dim(image, new_dim):
    # Determine current pixel spacing
    resize_factor = new_dim / np.array(image.shape)

    image = scipy.ndimage.interpolation.zoom(image, resize_factor, mode='nearest')

    return image

if __name__=="__main__":
  dataset_directory = '/data/mdstudents/Head-Neck-PET-CT'

  output_directory = os.path.join(dataset_directory+"_preprocessed")
  if not os.path.isdir(output_directory):
      os.makedirs(output_directory)

  patient_list = [os.path.join(dataset_directory, name) for name in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, name))]

  exclusion_list = []

  for patient in patient_list:
      if patient in exclusion_list: continue
      print(patient)
      for study in os.listdir(patient):
          if os.path.exists(os.path.join(output_directory, os.path.basename(patient), study, "CT.nii.gz")): continue
          for subdir, dirs, files in os.walk(os.path.join(patient, study)):
              dcms = glob.glob(os.path.join(subdir, "*.dcm"))
              if len(dcms) == 1:
                  structure = dicom.read_file(os.path.join(subdir, files[0]))

                  try:
                      series_description = structure.SeriesDescription
                  except AttributeError:
                      series_description = 'RTstruct_CTsim->CT(PET-CT)'
                  if series_description == 'RTstruct_CTsim->PET(PET-CT)':continue

                  contours = read_structure(structure)
                  print('RTStructure drawn: ', os.path.dirname(dcms[0]))
              elif len(dcms) > 1:
                  slices = [dicom.read_file(dcm) for dcm in dcms]
                  slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
                  if slices[0].Modality == 'CT':
                      image = get_pixels_hu(slices) #convert to hounsfeld unit
                      print('CT acquired: ', os.path.dirname(dcms[0]))
      #             elif slices[0].Modality == 'PT':
      #                pet_image = np.stack([s.pixel_array for s in slices], axis=-1)
      #                pet_image, new_spacing = resample(pet_image, slices)

          try:
              label, colors = get_mask(contours, slices) #colors can be used to visualize using pyplot
          except (IndexError, ValueError) as e:
              print(patient, e)
              shutil.move(os.path.join(patient, study), os.path.join('/data/mdstudents/tmp', os.path.basename(patient)))
              continue

          # resample to 1x1x1
          image, new_spacing = resample(image, slices)
          label = resample_to_dim(label, np.array(image.shape))

          #inspect the voxel dimensions for all scans
          spacing = np.array(list(slices[0].PixelSpacing) + [slices[0].SliceThickness], dtype=np.float32)
          print('spacing: ', spacing)

          if not os.path.isdir(os.path.join(output_directory, os.path.basename(patient))):
              os.makedirs(os.path.join(output_directory, os.path.basename(patient)))

          if not os.path.isdir(os.path.join(output_directory, os.path.basename(patient), study)):
              os.makedirs(os.path.join(output_directory, os.path.basename(patient), study))

          # Saves files in .nii.gz format
          image_nii = nib.Nifti1Image(image, np.eye(4))
          image_nii.to_filename(os.path.join(output_directory, os.path.basename(patient), study, "CT.nii.gz"))
          mask_nii = nib.Nifti1Image(label, np.eye(4))
          mask_nii.to_filename(os.path.join(output_directory, os.path.basename(patient), study, "mask.nii.gz"))

      # Saves files in .npy format
      #     np.save(os.path.join(output_directory, os.path.basename(patient), "CT.npy"), image)
      #     np.save(os.path.join(output_directory, os.path.basename(patient), "PET.npy"), pet_image)
      #     np.save(os.path.join(output_directory, os.path.basename(patient)+"_label.npy"), label)
