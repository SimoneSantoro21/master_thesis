import os
import re
import nibabel as nib
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchio as tio 
from skimage.transform import resize
import numpy as np

class DiffusionDataset(Dataset):
    """
    PyTorch dataset class for diffusion data with patient grouping and consistent resizing.
    Expected folder structure in the provided root directory:
      - CENTERS: Contains center slices in NIfTI format.
      - NEIGHBORS: Contains neighbor slices in NIfTI format.
    """

    def __init__(self, root_path, center_index, transform=None, resize_shape=(128, 128), norm='b0'):
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
        self.norm = norm

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
            patient_id, slice_num = index
            # Find the file in patient_center_files where the slice number matches
            center_list = self.patient_center_files.get(patient_id)
            neighbor_list = self.patient_neighbor_files.get(patient_id)

            if center_list is None or neighbor_list is None:
                raise KeyError(f"Patient {patient_id} not found in dataset.")

            # Search for the file whose slice number equals slice_num
            center_file = None

            for s, fname in center_list:
                if s == slice_num:
                    center_file = fname
                    break

            if center_file is None:
                raise IndexError(f"No center slice with number {slice_num} for patient {patient_id}.")
            

            neighbor_file = None
            for s, fname in neighbor_list:
                if s == slice_num:
                    neighbor_file = fname
                    break

            if neighbor_file is None:
                raise IndexError(f"No neighbor slice with number {slice_num} for patient {patient_id}.")

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

        # Normalize images to [0,1]
        if self.norm == 'minmax':
            center_min, center_max = center_arr.min(), center_arr.max()
            if center_max > center_min:
                center_arr = center_arr / 100
            else:
                center_arr = center_arr / 100

            neighbor_min, neighbor_max = neighbor_arr[3].min(), neighbor_arr[3].max()
            if neighbor_max > neighbor_min:
                neighbor_arr = neighbor_arr / 100
            else:
                neighbor_arr = neighbor_arr / 100

        elif self.norm == 'b0':
            b0_slice = neighbor_arr[:, :, 3]
            threshold = 1000
            binary_map = np.where(np.abs(b0_slice) < threshold, 0, 1)
            b0_slice = b0_slice * binary_map  # suppress near-zero B0 values

            for i in range(neighbor_arr.shape[2]):
                if i == 3:
                    # Skip B0, or alternatively:
                    neighbor_arr[:, :, i] = binary_map  # If you're setting it as a mask
                    continue

                neighbor_arr[:, :, i] *= binary_map
                np.divide(neighbor_arr[:, :, i], b0_slice, out=neighbor_arr[:, :, i], where=(b0_slice != 0))
            
            b0_3d = np.expand_dims(b0_slice, axis=-1)
            binary_map_3d = np.expand_dims(binary_map, axis=-1)

            center_arr *= binary_map_3d
            condition_3d = np.expand_dims(b0_slice != 0, axis=-1)
            np.divide(center_arr, b0_3d, out=center_arr, where=condition_3d)

            
        # Convert to tensors and permute to [D, H, W]
        center_tensor = torch.tensor(center_arr, dtype=torch.float32).permute(2, 0, 1)
        neighbor_tensor = torch.tensor(neighbor_arr, dtype=torch.float32).permute(2, 0, 1)

        # Add channel dimension [1, D, H, W]
        center_tensor = center_tensor.unsqueeze(0)
        neighbor_tensor = neighbor_tensor.unsqueeze(0)

        # Resize each slice in the volume to (self.resize_shape) using bicubic interpolation
        # Input shape: [1, D, H, W]
        center_slices = []
        for i in range(center_tensor.shape[1]):
            slice_i = center_tensor[0, i].numpy()
            resized = resize(slice_i, self.resize_shape, order=3, mode='reflect', preserve_range=True, anti_aliasing=True)
            center_slices.append(torch.tensor(resized, dtype=torch.float32))
        center_tensor = torch.stack(center_slices)  # [D, H, W]

        neighbor_slices = []
        for i in range(neighbor_tensor.shape[1]):
            slice_i = neighbor_tensor[0, i].numpy()
            resized = resize(slice_i, self.resize_shape, order=3, mode='reflect', preserve_range=True, anti_aliasing=True)
            neighbor_slices.append(torch.tensor(resized, dtype=torch.float32))
        neighbor_tensor = torch.stack(neighbor_slices)  # [D, H, W]

        # Add channel dimension -> [1, D, H, W]
        center_tensor = center_tensor.unsqueeze(0)
        neighbor_tensor = neighbor_tensor.unsqueeze(0)

        # Remove batch dimension -> [C, H, W]
        center_tensor = center_tensor.squeeze(0)
        neighbor_tensor = neighbor_tensor.squeeze(0)

        return neighbor_tensor, center_tensor
    