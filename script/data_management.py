import os
import re
import random
import numpy as np
import nibabel as nib
import pandas as pd
import ast


data_dir = '/Volumes/SEAGATE BAS/DTI_data'
base_dir = os.path.join(data_dir, 'base_dataset')

# Files that contains the patient indexes
patient_indexes_file = os.path.join(base_dir, "patient_indexes.txt")
training_indexes_file = os.path.join(base_dir, "training_patients.txt")
validation_indexes_file = os.path.join(base_dir, "validation_patients.txt")
testing_indexes_file = os.path.join(base_dir, "test_patients copy.txt")

with open(patient_indexes_file, "r") as f:
    patient_indexes = [line.strip() for line in f if line.strip()]

print("Found patient indexes:", patient_indexes)

with open(training_indexes_file, "r") as f:
    train_patients = [line.strip() for line in f if line.strip()]

print("Found training patients:", train_patients)

with open(validation_indexes_file, "r") as f:
    val_patients = [line.strip() for line in f if line.strip()]

print("Found validation patients:", val_patients)

with open(testing_indexes_file, "r") as f:
    test_patients = [line.strip() for line in f if line.strip()]

print("Found testing patients:", test_patients)

#manual_seed = 42
#random.seed(manual_seed)
#
## Randomly split into 80% training and 20% validation
#num_patients = len(patient_indexes)
#num_train = int(0.7 * num_patients)
#num_val = int(0.2 * num_patients)
#shuffled = patient_indexes.copy()
#random.shuffle(shuffled)
#train_patients = shuffled[:num_train]
#val_patients = shuffled[num_train:(num_train + num_val)]
#test_patients = shuffled[(num_train + num_val):]
#
## Save the training and validation lists outside the Dbs folders
#with open(os.path.join(base_dir, "training_patients.txt"), "w") as f:
#    for p in train_patients:
#        f.write(p + "\n")
#with open(os.path.join(base_dir, "validation_patients.txt"), "w") as f:
#    for p in val_patients:
#        f.write(p + "\n")
#with open(os.path.join(base_dir,"test_patients.txt"), "w") as f:
#    for p in test_patients:
#        f.write(p + "\n")
#
#print("Training patients:", train_patients)
#print("Validation patients:", val_patients)
#print("Test patients:", test_patients)

# find the diffusion file in a patient's Converted_Nii_Files_1 folder.
def find_diffusion_file(converted_folder):
    for file in os.listdir(converted_folder):
        lower_file = file.lower()
        if (
            "ax" in lower_file and
            "b__1000_multiband.nii.gz" in lower_file and
            not "rev" in lower_file  # optionally exclude reversed polarity images
        ):
            return os.path.join(converted_folder, file)
    return None


# Process a single patient folder.
def process_patient(patient, phase, center_index, output_root):
    # Construct patient folder path, e.g. base_dir/Dbs_[patient]
    patient_folder = os.path.join(base_dir, f"Dbs_{patient}")
    converted_folder = os.path.join(patient_folder, "Converted_Nii_Files_1")
    diffusion_file = os.path.join(converted_folder, 'b100_BET.nii.gz')
    if diffusion_file is None:
        print(f"Patient {patient}: Diffusion file not found in {converted_folder}. Skipping.")
        return

    print(f"Processing patient {patient} ({phase}): {diffusion_file}")

    # Load the 4D image
    img_nib = nib.load(diffusion_file)
    image_affine = img_nib.affine
    print(f"{diffusion_file} shape: {img_nib.shape}")

    # Extract the center volume for the given center index (assumed shape: X x Y x Z)
    center_volume = img_nib.slicer[:, :, :, center_index].get_fdata()

    #extract b0 volume
    b_0_volume = img_nib.slicer[:, :, :, 0].get_fdata()

    # Load the patient-specific CSV file from the patient folder.
    csv_path = '/Volumes/SEAGATE BAS/DTI_data/base_dataset/Dbs_001/dataframe.csv'
    if not os.path.exists(csv_path):
        print(f"Patient {patient}: CSV file not found in {patient_folder}. Skipping.")
        return
    df_patient = pd.read_csv(csv_path, converters={
        "Center Direction": lambda v: ast.literal_eval(v),
        "Neighbors directions": lambda v: ast.literal_eval(v),
        "Neighbor Indices": lambda v: ast.literal_eval(v)
    })

    # Extract neighbor indices for the given center index.
    try:
        df_row = df_patient[df_patient["Center Index"] == center_index].iloc[0]
    except IndexError:
        print(f"Patient {patient}: No row with Center Index {center_index} in CSV. Skipping.")
        return
    neighbor_indices = df_row["Neighbor Indices"]

    # Extract neighbor volumes for each neighbor direction
    neighbor_volumes = []
    for idx in neighbor_indices:
        if idx == -1:
            continue
        nvol = img_nib.slicer[:, :, :, idx].get_fdata()
        neighbor_volumes.append(nvol)
    # Use only the first three neighbors if available
    if neighbor_volumes:
        neighbor_volumes = neighbor_volumes[:2]
        neighbor_volumes.append(b_0_volume)
        neighbor_concat = np.stack(neighbor_volumes, axis=-1)
    else:
        neighbor_concat = None

    # Determine number of axial slices (e.g. 62 slices)
    num_slices = center_volume.shape[2]

    # Output directories for this phase
    centers_out = os.path.join(output_root, phase, "CENTERS")
    neighbors_out = os.path.join(output_root, phase, "NEIGHBORS")
    
    # For each axial slice, save center and neighbor slices.
    for slice_index in range(num_slices):
        # Extract center slice (2D array) and add a singleton channel dimension.
        center_slice = center_volume[:, :, slice_index]
        center_slice_3d = center_slice[..., np.newaxis]
        center_filename = os.path.join(centers_out, f"centered_{patient}_{center_index}_{slice_index}.nii.gz")
        center_img_nifti = nib.Nifti1Image(center_slice_3d, image_affine)
        nib.save(center_img_nifti, center_filename)
        
        # If neighbor volumes exist, extract corresponding slice.
        if neighbor_concat is not None:
            # neighbor_concat shape: (X, Y, num_slices, Nneighbors)
            neighbor_slice = neighbor_concat[:, :, slice_index, :]
            neighbor_filename = os.path.join(neighbors_out, f"neighbors_{patient}_{center_index}_{slice_index}.nii.gz")
            neighbor_img_nifti = nib.Nifti1Image(neighbor_slice, image_affine)
            nib.save(neighbor_img_nifti, neighbor_filename)
    print(f"Patient {patient} processed and slices saved.")

if __name__ == '__main__':
    # Global diffusion extraction parameters
    theta_max = 30  # degrees (for neighbor computation)
    theta_max_rad = np.deg2rad(theta_max)
    for center_index in range(1, 31):
        # Output folder
        output_root = os.path.join(data_dir, 'single_directions/2neighbours_extension', f"dataset_center{center_index}")
        for phase in ["TRAINING", "VALIDATION", "TEST"]:
            for subfolder in ["CENTERS", "NEIGHBORS"]:
                os.makedirs(os.path.join(output_root, phase, subfolder), exist_ok=True)
        #print("Processing TRAINING patients...")
        #for patient in train_patients:
        #    process_patient(patient, "TRAINING", center_index, output_root)

        #print("Processing VALIDATION patients...")
        #for patient in val_patients:
        #    process_patient(patient, "VALIDATION", center_index, output_root)

        print("Processing TEST patients...")
        for patient in test_patients:
            process_patient(patient, "TEST", center_index, output_root)


        print("All processing complete.")
