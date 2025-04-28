import os
import re
import nibabel as nib
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from skimage.transform import resize
import numpy as np

import os
import re
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

class DiffusionDataset(Dataset):
    """
    A unified dataset over ALL b-vec centre indices (1–30) in one folder.
    Expects two subfolders:
      - CENTERS:   centered_{patientid}_{center_idx}_{slice}.nii.gz
      - NEIGHBORS: neighbors_{patientid}_{center_idx}_{slice}.nii.gz
    """

    def __init__(self,
                 root_path: str,
                 transform=None,
                 resize_shape=(128, 128),
                 norm='b0'):
        self.centers_dir   = os.path.join(root_path, "CENTERS")
        self.neighbors_dir = os.path.join(root_path, "NEIGHBORS")
        self.transform     = transform
        self.resize_shape  = resize_shape
        self.norm          = norm

        # Regex to capture patient_id, center_idx (1–30), and slice_idx
        pat_center = re.compile(r'^centered_([^_]+)_(\d{1,2})_(\d+)\.nii\.gz$')
        pat_neighbor = re.compile(r'^neighbors_([^_]+)_(\d{1,2})_(\d+)\.nii\.gz$')

        # Map key=(center_idx, patient_id, slice_idx) -> filename
        center_map   = {}
        neighbor_map = {}

        for fname in os.listdir(self.centers_dir):
            m = pat_center.match(fname)
            if not m: continue
            pid, ci, sl = m.group(1), int(m.group(2)), int(m.group(3))
            center_map[(ci, pid, sl)] = fname

        for fname in os.listdir(self.neighbors_dir):
            m = pat_neighbor.match(fname)
            if not m: continue
            pid, ci, sl = m.group(1), int(m.group(2)), int(m.group(3))
            neighbor_map[(ci, pid, sl)] = fname

        # Build samples list (only keys present in both maps)
        self.samples = []
        for key in center_map:
            if key in neighbor_map:
                cfile = os.path.join(self.centers_dir,   center_map[key])
                nfile = os.path.join(self.neighbors_dir, neighbor_map[key])
                self.samples.append((key[0], key[1], key[2], cfile, nfile))

        # sort by center_idx then patient then slice
        self.samples.sort(key=lambda x: (x[0], x[1], x[2]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # unpack sample
        center_idx, patient_id, slice_idx, cpath, npath = self.samples[idx]

        center_arr   = nib.load(cpath).get_fdata()
        neighbor_arr = nib.load(npath).get_fdata()

        if self.transform:
            center_arr   = self.transform(center_arr)
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
            b0_slice = b0_slice * binary_map 

            for i in range(neighbor_arr.shape[2]):
                if i == 3:
                    neighbor_arr[:, :, i] = binary_map 
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

        # Resize each slice in the volume
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