#!/usr/bin/env python3
import os
import shutil


def merge_dataset(src_root: str, dst_root: str):
    """
    Copy contents of CENTERS/ and NEIGHBORS/ under TRAINING, TEST and VALIDATION
    from src_root into dst_root
    """
    splits = ["TRAINING", "TEST", "VALIDATION"]
    subdirs = ["CENTERS", "NEIGHBORS"]

    for split in splits:
        for sub in subdirs:
            src_dir = os.path.join(src_root, split, sub)
            dst_dir = os.path.join(dst_root, split, sub)

            if not os.path.isdir(src_dir):
                print(f"[!] Source directory not found, skipping: {src_dir}")
                continue

            os.makedirs(dst_dir, exist_ok=True)

            files = os.listdir(src_dir)
            for fname in files:
                src_file = os.path.join(src_dir, fname)
                dst_file = os.path.join(dst_dir, fname)
                shutil.copy2(src_file, dst_file)

            print(f"Copied {len(files)} files → {dst_dir}")

if __name__ == "__main__":
    DST_ROOT = '/Volumes/SEAGATE BAS/DTI_data/multishell_b2600'
    for index in range(1, 114):
        SRC_ROOT = f"/Volumes/SEAGATE BAS/DTI_data/multishell_b2600_single_directions/dataset_ms_center{index}"
        print(f"Merging from `{SRC_ROOT}` into `{DST_ROOT}`…")
        merge_dataset(SRC_ROOT, DST_ROOT)
        print("Done.")