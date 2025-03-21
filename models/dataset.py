import os
import re
import nibabel as nib
from torch.utils.data import Dataset
import torch

class DiffusionDataset(Dataset):
    """
    PyTorch dataset class for diffusion data with patient grouping.

    Expected folder structure in the provided root directory:
      - CENTERS: Contains center slices in NIfTI format.
                 Files must be named as "centered_<patient_index>_<center_index>_<slice_number>.nii.gz".
      - NEIGHBORS: Contains neighbor slices in NIfTI format.
                   Files must be named as "neighbors_<patient_index>_<center_index>_<slice_number>.nii.gz".

    The dataset is built for a specified center direction (center_index). Each sample corresponds 
    to one axial slice from a specific patient.
    
    The __getitem__ method supports two kinds of indexing:
      - Integer indexing: The dataset creates an internal list of (patient_id, slice_idx) pairs.
      - Tuple indexing: You can directly pass (patient_id, slice_idx) to select the desired sample.
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

        # Regex patterns to capture patient index and slice number.
        # For centers: "centered_<patient_index>_<center_index>_<slice_number>.nii.gz"
        pattern_center = re.compile(r'^centered_([^_]+)_{}_(\d+)\.nii\.gz$'.format(center_index))
        # For neighbors: "neighbors_<patient_index>_<center_index>_<slice_number>.nii.gz"
        pattern_neighbor = re.compile(r'^neighbors_([^_]+)_{}_(\d+)\.nii\.gz$'.format(center_index))
        
        # Group center files by patient index.
        self.patient_center_files = {}
        for f in os.listdir(self.centers_dir):
            m = pattern_center.match(f)
            if m:
                pat_id = m.group(1)
                slice_num = int(m.group(2))
                self.patient_center_files.setdefault(pat_id, []).append((slice_num, f))
        
        # Group neighbor files by patient index.
        self.patient_neighbor_files = {}
        for f in os.listdir(self.neighbors_dir):
            m = pattern_neighbor.match(f)
            if m:
                pat_id = m.group(1)
                slice_num = int(m.group(2))
                self.patient_neighbor_files.setdefault(pat_id, []).append((slice_num, f))
        
        # Sort files for each patient by slice number.
        for pat_id in self.patient_center_files:
            self.patient_center_files[pat_id] = sorted(self.patient_center_files[pat_id], key=lambda x: x[0])
        for pat_id in self.patient_neighbor_files:
            self.patient_neighbor_files[pat_id] = sorted(self.patient_neighbor_files[pat_id], key=lambda x: x[0])
        
        # Ensure that each patient has matching numbers of center and neighbor slices.
        for pat_id in self.patient_center_files:
            if pat_id not in self.patient_neighbor_files:
                raise ValueError(f"Missing neighbor data for patient {pat_id}.")
            if len(self.patient_center_files[pat_id]) != len(self.patient_neighbor_files[pat_id]):
                raise ValueError(f"Mismatch in number of slices for patient {pat_id}.")

        # Build an internal list of (patient_id, slice_index) pairs for integer indexing.
        self.keys = []
        for pat_id, center_list in self.patient_center_files.items():
            for i in range(len(center_list)):
                self.keys.append((pat_id, i))
                
    def __len__(self):
        """
        Returns:
            int: Total number of slices (samples) across all patients.
        """
        return len(self.keys)
    
    def __getitem__(self, index):
        """
        Loads and returns the neighbor and center slice for a given (patient, slice) pair.

        Parameters:
            index (int or tuple): If an integer, it is used to index into an internal list of 
                                  (patient_id, slice_index) pairs. If a tuple, it should be 
                                  (patient_id, slice_index).

        Returns:
            tuple: (neighbor_slice, center_slice) as PyTorch tensors.
        """
        # Support both tuple indexing and integer indexing.
        if isinstance(index, tuple):
            patient_id, slice_idx = index
        else:
            patient_id, slice_idx = self.keys[index]
        
        # Get file names for the given patient and slice index.
        center_tuple = self.patient_center_files.get(patient_id)
        neighbor_tuple = self.patient_neighbor_files.get(patient_id)
        if center_tuple is None or neighbor_tuple is None:
            raise ValueError(f"No data available for patient {patient_id}.")
        if slice_idx < 0 or slice_idx >= len(center_tuple):
            raise IndexError(f"Slice index {slice_idx} out of range for patient {patient_id}.")

        center_filename = center_tuple[slice_idx][1]
        neighbor_filename = neighbor_tuple[slice_idx][1]
        
        center_path = os.path.join(self.centers_dir, center_filename)
        neighbor_path = os.path.join(self.neighbors_dir, neighbor_filename)
        
        # Load images using nibabel.
        center_img = nib.load(center_path)
        neighbor_img = nib.load(neighbor_path)
        
        # Extract numpy arrays from the NIfTI images.
        center_arr = center_img.get_fdata()
        neighbor_arr = neighbor_img.get_fdata()
        
        # Optionally apply a transform.
        if self.transform is not None:
            center_arr = self.transform(center_arr)
            neighbor_arr = self.transform(neighbor_arr)
        
        # Convert arrays to PyTorch tensors.
        center_tensor = torch.tensor(center_arr, dtype=torch.float32)
        neighbor_tensor = torch.tensor(neighbor_arr, dtype=torch.float32)
        
        # Rearrange dimensions: assume the slices are in the third dimension.
        center_tensor = center_tensor.permute(2, 0, 1)
        neighbor_tensor = neighbor_tensor.permute(2, 0, 1)
        
        # Optionally add an extra channel dimension (if needed by your model).
        center_tensor = center_tensor.unsqueeze(0)
        neighbor_tensor = neighbor_tensor.unsqueeze(0)
        
        return neighbor_tensor, center_tensor
