import os
import numpy as np
import pandas as pd
import ast


# Base directory that contains the Dbs folders
base_path = '/Volumes/SEAGATE BAS/DTI_data/base_dataset'
INDEXES = os.path.join(base_path, 'patient_indexes.txt')

# Read patient indexes from the file "patient_indexes.txt"
with open(INDEXES, "r") as f:
    patient_indexes = [line.strip() for line in f if line.strip()]


# Define the maximum angular threshold (in degrees converted to radians)
theta_max = 30  # degrees
theta_max_rad = np.deg2rad(theta_max)

# Helper function to convert string representations to Python objects
def convert_complex(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return value

# Loop over each patient index
for patient in patient_indexes:
    # Construct the patient folder path
    patient_folder = os.path.join(base_path, f"Dbs_{patient}")
    # The Converted_Nii_Files_1 subfolder is assumed to be inside the patient folder
    converted_folder = os.path.join(patient_folder, "Converted_Nii_Files_1")
    
    # Search for the bvec and bval files in the converted_folder
    bvec_file = None
    bval_file = None
    for file in os.listdir(converted_folder):
        lower_file = file.lower()
        if "ax" in lower_file and "b__1000_multiband.bvec" in lower_file:
            bvec_file = os.path.join(converted_folder, file)
        if "ax" in lower_file and "b__1000_multiband.bval" in lower_file:
            bval_file = os.path.join(converted_folder, file)
            
    if bvec_file is None or bval_file is None:
        print(f"Patient {patient}: Could not find required bvec or bval files in {converted_folder}. Skipping...")
        continue

    # Load the bvec and bval files
    bvecs_single_shell = np.loadtxt(bvec_file)
    bvals_single_shell = np.loadtxt(bval_file)
    
    # Extract components for single-shell and compute reverse polarity vectors.
    # Assuming bvecs are organized as 3 rows.
    x_single, y_single, z_single = bvecs_single_shell
    x_reverse, y_reverse, z_reverse = -x_single, -y_single, -z_single
    
    # Combine original and reverse polarity directions
    x_combined = np.concatenate((x_single, x_reverse))
    y_combined = np.concatenate((y_single, y_reverse))
    z_combined = np.concatenate((z_single, z_reverse))
    
    # Initialize an empty list to store DataFrame rows
    data = []
    
    # Loop through each single-shell direction (skip index 0 if desired)
    for i in range(1, len(x_single)):
        solid_angle_center = np.array([x_single[i], y_single[i], z_single[i]])
        dot_products = x_combined * solid_angle_center[0] + \
                       y_combined * solid_angle_center[1] + \
                       z_combined * solid_angle_center[2]
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
        tolerance = 1e-6
        neighbor_indices_combined = np.where((angles <= theta_max_rad) & (angles > tolerance))[0]
        count_inside = len(neighbor_indices_combined)
        neighbors_directions = [(x_combined[j], y_combined[j], z_combined[j])
                                for j in neighbor_indices_combined]
        neighbor_indices_single = []
        for j in neighbor_indices_combined:
            neighbor_dir = np.array([x_combined[j], y_combined[j], z_combined[j]])
            found_index = None
            for k in range(len(x_single)):
                candidate = np.array([x_single[k], y_single[k], z_single[k]])
                if np.allclose(neighbor_dir, candidate, atol=1e-6):
                    found_index = k
                    break
            if found_index is None:
                reversed_dir = -neighbor_dir
                for k in range(len(x_single)):
                    candidate = np.array([x_single[k], y_single[k], z_single[k]])
                    if np.allclose(reversed_dir, candidate, atol=1e-6):
                        found_index = k
                        break
            neighbor_indices_single.append(found_index)
        neighbor_indices_single_tuple = tuple(neighbor_indices_single)
        data.append({
            "Solid Angle": theta_max_rad,
            "Center Direction": tuple(solid_angle_center),
            "Center Index": i,
            "# Neighbors": count_inside,
            "Neighbors directions": neighbors_directions,
            "Neighbor Indices": neighbor_indices_single_tuple
        })
    
    # Create DataFrame and save as CSV in the patient folder
    df = pd.DataFrame(data)
    print(f"Patient {patient} DataFrame:")
    print(df)
    csv_save_path = os.path.join(patient_folder, "dataframe.csv")
    df.to_csv(csv_save_path, index=False)
    print(f"Saved dataframe CSV for patient {patient} at {csv_save_path}\n")
