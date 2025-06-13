import os
import re
import random
import numpy as np
import nibabel as nib
import pandas as pd
import ast

# ---------------------------------
# Configuration and helper funcs
# ---------------------------------

data_dir = '/Volumes/SEAGATE BAS/DTI_data'
base_dir = os.path.join(data_dir, 'base_dataset')

# Lists of patient indexes
patient_indexes_file    = os.path.join(base_dir, "patients_multishell.txt")

def read_list(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

patient_indexes = read_list(patient_indexes_file)

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
#with open(os.path.join(base_dir, "training_patients_multishell.txt"), "w") as f:
#    for p in train_patients:
#        f.write(p + "\n")
#with open(os.path.join(base_dir, "validation_patients_multishell.txt"), "w") as f:
#    for p in val_patients:
#        f.write(p + "\n")
#with open(os.path.join(base_dir,"test_patients_multishell.txt"), "w") as f:
#    for p in test_patients:
#        f.write(p + "\n")

train_patients   = read_list(os.path.join(base_dir, "training_patients_multishell.txt"))
val_patients     = read_list(os.path.join(base_dir, "validation_patients_multishell.txt"))
test_patients    = read_list(os.path.join(base_dir, "test_patients_multishell.txt"))

print("Training patients:", train_patients)
print("Validation patients:", val_patients)
print("Test patients:", test_patients)

# Angular threshold
theta_max     = 30  # degrees
theta_max_rad = np.deg2rad(theta_max)

def find_file(folder, keyword):
    """Return the first filename in `folder` whose lowercase name includes `keyword`."""
    for fn in os.listdir(folder):
        if keyword in fn.lower():
            return os.path.join(folder, fn)
    return None

def process_patient(patient, phase, center_index, output_root):
    pt_folder   = os.path.join(base_dir, f"Dbs_{patient}")
    conv_folder = os.path.join(pt_folder, "Converted_Nii_Files_1")

    # Paths
    single_path = find_file(conv_folder, 'b100_bet.nii.gz')
    multi_path  = find_file(conv_folder, 'multishell_registered.nii.gz')
    csv_path    = os.path.join(pt_folder, "multishell_neighbors.csv")

    if not single_path or not multi_path or not os.path.exists(csv_path):
        print(f"[{phase}] Patient {patient}: missing single/multi NIfTI or CSV → skipping")
        return

    print(f"[{phase}] Patient {patient}:")
    print("  single-shell:", os.path.basename(single_path))
    print("  multi-shell :", os.path.basename(multi_path))
    print("  neighbor CSV:", os.path.basename(csv_path))

    # Load NIfTIs
    single_nib = nib.load(single_path)
    multi_nib  = nib.load(multi_path)

    # b0 volumes
    b0_single = single_nib.slicer[:, :, :, 0].get_fdata()
    b0_multi  = multi_nib.slicer[:, :, :, 0].get_fdata()

    # Load CSV
    df = pd.read_csv(csv_path, converters={
        "Center Direction": lambda v: ast.literal_eval(v),
        "Neighbors directions": lambda v: ast.literal_eval(v),
        "Neighbor Indices": lambda v: ast.literal_eval(v)
    })

    try:
        row = df[df["Center Index"] == center_index].iloc[0]
    except IndexError:
        print(f"  → no CSV row for Center Index {center_index}; skipping")
        return

    neigh_idxs = row["Neighbor Indices"]

    # --- Center stack: center + multishell b0 ---
    center_vol = multi_nib.slicer[:, :, :, center_index].get_fdata()
    center_concat = np.stack([center_vol, b0_multi], axis=-1)

    # --- Neighbor stack: neighbors + singleshell b0 ---
    neigh_vols = []
    for ni in neigh_idxs:
        if ni < 0:
            continue
        neigh_vols.append(single_nib.slicer[:, :, :, ni].get_fdata())
    neigh_vols = neigh_vols[:3]
    neigh_vols.append(b0_single)
    neigh_concat = np.stack(neigh_vols, axis=-1)

    # Prepare output
    center_out   = os.path.join(output_root, phase, "CENTERS")
    neighbor_out = os.path.join(output_root, phase, "NEIGHBORS")

    num_slices = center_vol.shape[2]
    for sl in range(num_slices):
        c2d = center_concat[:, :, sl, :]
        fn_c = os.path.join(center_out, f"center_{patient}_{center_index}_{sl}.nii.gz")
        nib.save(nib.Nifti1Image(c2d, multi_nib.affine), fn_c)

        n2d = neigh_concat[:, :, sl, :]
        fn_n = os.path.join(neighbor_out, f"neighbors_{patient}_{center_index}_{sl}.nii.gz")
        nib.save(nib.Nifti1Image(n2d, single_nib.affine), fn_n)

    print(f"  → done, saved {num_slices} slices each for center & neighbors\n")


if __name__ == "__main__":
    data_dir = '/Volumes/SEAGATE BAS/DTI_data'
    base_dir = os.path.join(data_dir, 'base_dataset')

    # Path to your master multishell CSV
    master_csv = os.path.join(base_dir, 'Dbs_001/multishell_neighbors.csv')
    df_all = pd.read_csv(master_csv)

    # Grab the unique list of center indices (zero-based)
    #center_indices = np.unique(df_all["Center Index"].to_numpy())
    #b500 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    #b1000 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 
    #b2600 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113
    center_indices = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113]
    print(f"Found {len(center_indices)} center indices: {center_indices}")


    for center_index in center_indices:
        out_root = os.path.join(data_dir,
                                'multishell_b2600_single_directions',
                                f"dataset_ms_center{center_index}")
        # Build folder structure
        for phase in ("TRAINING", "VALIDATION", "TEST"):
            for sub in ("CENTERS","NEIGHBORS"):
                os.makedirs(os.path.join(out_root, phase, sub), exist_ok=True)

        print(f"\n=== CENTER_INDEX = {center_index} ===")
        print("Processing TRAINING...")
        for p in train_patients:
            process_patient(p, "TRAINING", center_index, out_root)

        print("Processing VALIDATION...")
        for p in val_patients:
            process_patient(p, "VALIDATION", center_index, out_root)

        print("Processing TEST...")
        for p in test_patients:
            process_patient(p, "TEST", center_index, out_root)

    print("All done.")