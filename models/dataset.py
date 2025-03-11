import os
import re
import nibabel as nib
from torch.utils.data import Dataset
import torch


class DiffusionDataset(Dataset):
    """
    PyTorch dataset class for diffusion data.

    This dataset expects the following folder structure inside the provided root directory:
      - CENTERS: Contains center slices in NIfTI format.
                 Files must be named as "centered_<center_index>_<slice_index>.nii.gz".
      - NEIGHBORS: Contains neighbor slices (with channels) in NIfTI format.
                   Files must be named as "neighbors_<center_index>_<slice_index>.nii.gz".

    The dataset is built for a specified center direction (center_index). Each sample corresponds 
    to one axial slice.
    """
    
    def __init__(self, root_path, center_index, transform=None):
        """
        Initialize the dataset.

        Parameters:
            root_path (str): Root directory containing "CENTERS" and "NEIGHBORS" subfolders.
            center_index (int): The center direction index (e.g., 5) used in the filenames.
            transform (callable, optional): Optional transformation to apply to the images.
        """
        self.root_path = root_path
        self.center_index = center_index
        self.transform = transform

        self.centers_dir = os.path.join(root_path, "CENTERS")
        self.neighbors_dir = os.path.join(root_path, "NEIGHBORS")

        # List and sort center files matching the naming convention.
        self.center_files = sorted(
            [f for f in os.listdir(self.centers_dir)
             if f.startswith(f"centered_{center_index}_") and f.endswith(".nii.gz")],
            key=lambda x: int(re.findall(r'centered_{}_([0-9]+)'.format(center_index), x)[0])
        )

        # List and sort neighbor files.
        self.neighbor_files = sorted(
            [f for f in os.listdir(self.neighbors_dir)
             if f.startswith(f"neighbors_{center_index}_") and f.endswith(".nii.gz")],
            key=lambda x: int(re.findall(r'neighbors_{}_([0-9]+)'.format(center_index), x)[0])
        )

        if len(self.center_files) != len(self.neighbor_files):
            raise ValueError("Mismatch between number of center slices and neighbor slices.")

    def __len__(self):
        """
        Returns:
            int: Total number of slices (samples) for the specified center direction.
        """
        return len(self.center_files)

    def __getitem__(self, index):
        """
        Loads and returns the center and neighbor slice for a given sample index.

        Parameters:
            index (int): Index of the slice (0-indexed).

        Returns:
            tuple: (center_slice, neighbor_slice) as PyTorch tensors.
        """
        center_path = os.path.join(self.centers_dir, self.center_files[index])
        neighbor_path = os.path.join(self.neighbors_dir, self.neighbor_files[index])

        # Load images using nibabel
        center_img = nib.load(center_path)
        neighbor_img = nib.load(neighbor_path)

        # Extract numpy arrays from the NIfTI images.
        center_arr = center_img.get_fdata()  
        neighbor_arr = neighbor_img.get_fdata()  

        # Optionally apply a transform to the arrays.
        if self.transform is not None:
            center_arr = self.transform(center_arr)
            neighbor_arr = self.transform(neighbor_arr)

        # Convert arrays to PyTorch tensors.
        center_tensor = torch.tensor(center_arr, dtype=torch.float32)
        center_tensor = center_tensor.permute(2, 0, 1)
        center_tensor.unsqueeze(0)

        neighbor_tensor = torch.tensor(neighbor_arr, dtype=torch.float32)
        neighbor_tensor = neighbor_tensor.permute(2, 0, 1)
        neighbor_tensor.unsqueeze(0)

        return neighbor_tensor, center_tensor
