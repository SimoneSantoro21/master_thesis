import os
import re
import nibabel as nib
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class DiffusionDataset(Dataset):
    """
    PyTorch dataset class for diffusion data with patient grouping and consistent resizing.

    Expected folder structure in the provided root directory:
      - CENTERS: Contains center slices in NIfTI format.
      - NEIGHBORS: Contains neighbor slices in NIfTI format.
    """

    def __init__(self, root_path, center_index, transform=None, resize_shape=(128, 128)):
        """
        Initialize the dataset.

        Parameters:
            root_path (str): Root directory containing "CENTERS" and "NEIGHBORS" subfolders.
            center_index (int): The center direction index (e.g., 5) used in the filenames.
            transform (callable, optional): Optional transformation to apply to the images.
            resize_shape (tuple): Target (height, width) shape to resize all images.
        """
        self.root_path = root_path
        self.center_index = center_index
        self.transform = transform
        self.resize_shape = resize_shape

        self.centers_dir = os.path.join(root_path, "CENTERS")
        self.neighbors_dir = os.path.join(root_path, "NEIGHBORS")

        # Regex patterns for file matching
        pattern_center = re.compile(r'^centered_([^_]+)_{}_(\d+)\.nii\.gz$'.format(center_index))
        pattern_neighbor = re.compile(r'^neighbors_([^_]+)_{}_(\d+)\.nii\.gz$'.format(center_index))
        
        self.patient_center_files = {}
        self.patient_neighbor_files = {}

        for f in os.listdir(self.centers_dir):
            m = pattern_center.match(f)
            if m:
                pat_id, slice_num = m.group(1), int(m.group(2))
                self.patient_center_files.setdefault(pat_id, []).append((slice_num, f))

        for f in os.listdir(self.neighbors_dir):
            m = pattern_neighbor.match(f)
            if m:
                pat_id, slice_num = m.group(1), int(m.group(2))
                self.patient_neighbor_files.setdefault(pat_id, []).append((slice_num, f))

        # Sort slice filenames by index
        for pat_id in self.patient_center_files:
            self.patient_center_files[pat_id] = sorted(self.patient_center_files[pat_id], key=lambda x: x[0])
        for pat_id in self.patient_neighbor_files:
            self.patient_neighbor_files[pat_id] = sorted(self.patient_neighbor_files[pat_id], key=lambda x: x[0])

        # Sanity check
        for pat_id in self.patient_center_files:
            if pat_id not in self.patient_neighbor_files:
                raise ValueError(f"Missing neighbor data for patient {pat_id}.")
            if len(self.patient_center_files[pat_id]) != len(self.patient_neighbor_files[pat_id]):
                raise ValueError(f"Mismatch in number of slices for patient {pat_id}.")

        self.keys = [(pat_id, i) for pat_id in self.patient_center_files for i in range(len(self.patient_center_files[pat_id]))]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            patient_id, slice_idx = index
        else:
            patient_id, slice_idx = self.keys[index]

        center_file = self.patient_center_files[patient_id][slice_idx][1]
        neighbor_file = self.patient_neighbor_files[patient_id][slice_idx][1]

        center_path = os.path.join(self.centers_dir, center_file)
        neighbor_path = os.path.join(self.neighbors_dir, neighbor_file)

        center_arr = nib.load(center_path).get_fdata()
        neighbor_arr = nib.load(neighbor_path).get_fdata()

        if self.transform:
            center_arr = self.transform(center_arr)
            neighbor_arr = self.transform(neighbor_arr)

        # Normalize images to have values between 0 and 1
        center_min, center_max = center_arr.min(), center_arr.max()
        if center_max > center_min:
            center_arr = (center_arr - center_min) / (center_max - center_min)
        else:
            center_arr = center_arr - center_min  # if constant, set to 0

        neighbor_min, neighbor_max = neighbor_arr.min(), neighbor_arr.max()
        if neighbor_max > neighbor_min:
            neighbor_arr = (neighbor_arr - neighbor_min) / (neighbor_max - neighbor_min)
        else:
            neighbor_arr = neighbor_arr - neighbor_min

        # Convert to tensors
        center_tensor = torch.tensor(center_arr, dtype=torch.float32).permute(2, 0, 1)  # [D, H, W]
        neighbor_tensor = torch.tensor(neighbor_arr, dtype=torch.float32).permute(2, 0, 1)

        # Add channel dim: [1, D, H, W]
        center_tensor = center_tensor.unsqueeze(0)
        neighbor_tensor = neighbor_tensor.unsqueeze(0)

        # Resize to target shape (e.g. [1, D, 128, 128])
        center_tensor = F.interpolate(center_tensor, size=self.resize_shape, mode='bilinear', align_corners=False)
        neighbor_tensor = F.interpolate(neighbor_tensor, size=self.resize_shape, mode='bilinear', align_corners=False)

        # Remove batch dim: [C, H, W]
        center_tensor = center_tensor.squeeze(0)
        neighbor_tensor = neighbor_tensor.squeeze(0)

        return neighbor_tensor, center_tensor
