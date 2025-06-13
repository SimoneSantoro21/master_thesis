import os
import numpy as np
import pandas as pd

# Base directory that contains the Dbs folders
base_path = '/Volumes/SEAGATE BAS/DTI_data/base_dataset'
INDEXES = os.path.join(base_path, 'patient_indexes.txt')

# Read patient indexes
with open(INDEXES, "r") as f:
    patient_indexes = [line.strip() for line in f if line.strip()]

theta_max     = 30  # degrees
theta_max_rad = np.deg2rad(theta_max)

found_patients = []

for patient in patient_indexes:
    patient_folder = os.path.join(base_path, f"Dbs_{patient}")
    conv = os.path.join(patient_folder, "Converted_Nii_Files_1")

    # find single-shell and multishell bvec/bval files
    bvec_ss = bval_ss = bvec_ms = bval_ms = None
    for fn in os.listdir(conv):
        low = fn.lower()
        if "ax" in low and "b__1000_multiband.bvec" in low:
            bvec_ss = os.path.join(conv, fn)
        if "ax" in low and "b__1000_multiband.bval" in low:
            bval_ss = os.path.join(conv, fn)
        if "ax" in low and "multishell_multiband.bvec" in low:
            bvec_ms = os.path.join(conv, fn)
        if "ax" in low and "multishell_multiband.bval" in low:
            bval_ms = os.path.join(conv, fn)

    if not all([bvec_ss, bval_ss, bvec_ms, bval_ms]):
        print(f"Patient {patient}: missing files, skipping.")
        continue

    found_patients.append(patient)

    # Load single-shell (keep b0 here, we'll ignore in neighbor selection)
    bvecs_single = np.loadtxt(bvec_ss)   # shape (3, Nss)
    bvals_single = np.loadtxt(bval_ss)   # shape (Nss,)
    x_ss, y_ss, z_ss = bvecs_single

    # Build combined single-shell arrays (polarity) and combined bvals
    x_comb     = np.concatenate((x_ss,   -x_ss))
    y_comb     = np.concatenate((y_ss,   -y_ss))
    z_comb     = np.concatenate((z_ss,   -z_ss))
    bvals_comb = np.concatenate((bvals_single, bvals_single))

    # Load multishell (keep b0 in arrays, we'll skip zero bvals in loop)
    bvecs_multi = np.loadtxt(bvec_ms)    # shape (3, Nms)
    bvals_multi = np.loadtxt(bval_ms)    # shape (Nms,)
    x_m_all, y_m_all, z_m_all = bvecs_multi

    data = []
    Nss = len(x_ss)
    for i in range(len(bvals_multi)):
        bval_m = bvals_multi[i]
        if bval_m == 0:
            # skip b0s but keep indexing unchanged
            continue

        center = np.array([x_m_all[i], y_m_all[i], z_m_all[i]])

        # compute angles to every single-shell ±
        dots   = x_comb*center[0] + y_comb*center[1] + z_comb*center[2]
        angles = np.arccos(np.clip(dots, -1.0, 1.0))

        tol = 1e-6
        # only pick neighbors within threshold and exclude any b0 neighbors
        idxs = np.where(
            (angles <= theta_max_rad) &
            (angles > tol) &
            (bvals_comb != 0)
        )[0]

        neigh_dirs  = [(x_comb[j], y_comb[j], z_comb[j]) for j in idxs]
        neigh_count = len(idxs)

        # map back to single-shell index (0..Nss-1)
        neigh_idx_ss = tuple((j if j < Nss else j - Nss) for j in idxs)

        data.append({
            "Multi_Bval":            bval_m,
            "Solid Angle":           theta_max_rad,
            "Center Direction":      tuple(center),
            "Center Index":          i,
            "# Neighbors":           neigh_count,
            "Neighbors directions":  neigh_dirs,
            "Neighbor Indices":      neigh_idx_ss
        })

    df = pd.DataFrame(data)
    out_csv = os.path.join(patient_folder, "multishell_neighbors.csv")
    df.to_csv(out_csv, index=False)
    print(f"Patient {patient}: wrote {len(df)} rows to {out_csv}")

# Save the list of patients with multishell data
out_list = os.path.join(base_path, "patients_multishell.txt")
with open(out_list, "w") as f:
    for p in found_patients:
        f.write(p + "\n")

print(f"\nTotal patients with multishell data: {len(found_patients)}")
print(f"Patient list saved to: {out_list}")
