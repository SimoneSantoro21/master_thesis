import os
import numpy as np
import nibabel as nib

input_folder = '/Volumes/SEAGATE BAS/DTI_data/output_patient_022'
reference_image_path = '/Volumes/SEAGATE BAS/DTI_data/base_dataset/Dbs_003/Converted_Nii_Files_1/b100_BET.nii.gz'
output_path = os.path.join(input_folder, 'final_image.nii.gz')

# Target shape (spatial dimensions) and number of volumes (excluding the reference b0)
target_shape = (128, 128, 62)
num_volumes = 30

# Load the reference image and extract its components.
ref_img = nib.load(reference_image_path)
ref_data = ref_img.get_fdata()
ref_affine = ref_img.affine
ref_header = ref_img.header.copy()

ref_volume = ref_data[:, :, :, 0]


if ref_volume.shape != target_shape:
    raise ValueError(f"Reference volume shape {ref_volume.shape} does not match target shape {target_shape}.")


# Expand the normalized b0 volume to 4D
first_vol_4d = ref_volume[..., np.newaxis]

# Pre-allocate for the generated volumes
image_4d = np.zeros((*target_shape, num_volumes), dtype=np.float32)

# Load and process each generated slice
for vol_idx in range(1, num_volumes + 1):
    for slice_idx in range(target_shape[2]):
        filename = f'output_003_center{vol_idx}_slice{slice_idx}.nii.gz'
        filepath = os.path.join(input_folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing file: {filename}")

        img = nib.load(filepath)
        slice_data = img.get_fdata()

        b0_slice = first_vol_4d[:, :, slice_idx, 0]
        slice_data *= b0_slice

        if slice_data.shape != (target_shape[0], target_shape[1]):
            raise ValueError(f"Unexpected slice shape {slice_data.shape} for file {filename}.")

        image_4d[:, :, slice_idx, vol_idx - 1] = slice_data

# Concatenate b0 with generated volumes
image_4d_with_b0 = np.concatenate((first_vol_4d, image_4d), axis=3)

#mask_filename = '/Volumes/SEAGATE BAS/DTI_data/base_dataset/Dbs_003/Converted_Nii_Files_1/BET_DTI_mask.nii.gz'
#mask_img = nib.load(mask_filename)
#mask_data = mask_img.get_fdata()
#
## Multiply the mask with each volume of the original 4D DTI data.
#masked_dti_data = image_4d_with_b0 * mask_data[..., np.newaxis]

# Update header
new_header = ref_header.copy()
new_header.set_data_shape(image_4d_with_b0.shape)

final_img = nib.Nifti1Image(image_4d_with_b0, affine=ref_affine, header=new_header)
nib.save(final_img, output_path)

print(f"Saved 4D image with thresholded, normalized slices and b0 to {output_path}")


