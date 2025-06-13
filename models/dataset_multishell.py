import os
import re
import nibabel as nib
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from skimage.transform import resize
import numpy as np
from models.metadata_encoder import metadata_encoder  # Make sure this file is available

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

        self.get_metadata = metadata_encoder(csv_path='/data/people/jamesgrist/Desktop/multishell_all_directions/multishell_neighbors.csv')

        pat_center = re.compile(r'^center_([^_]+)_(\d{1,2})_(\d+)\.nii\.gz$')
        pat_neighbor = re.compile(r'^neighbors_([^_]+)_(\d{1,2})_(\d+)\.nii\.gz$')

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

        self.samples = []
        for key in center_map:
            if key in neighbor_map:
                cfile = os.path.join(self.centers_dir,   center_map[key])
                nfile = os.path.join(self.neighbors_dir, neighbor_map[key])
                self.samples.append((key[0], key[1], key[2], cfile, nfile))

        self.samples.sort(key=lambda x: (x[0], x[1], x[2]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        center_idx, patient_id, slice_idx, cpath, npath = self.samples[idx]

        center_arr   = nib.load(cpath).get_fdata()  # shape: (H, W, 2)
        neighbor_arr = nib.load(npath).get_fdata()

        if self.transform:
            center_arr   = self.transform(center_arr)
            neighbor_arr = self.transform(neighbor_arr)

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
        # --- Normalize neighbors relative to single-shell b0 (channel 3) ---
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

        # --- Normalize center (multishell): channel 0 / channel 1, then discard b0 ---
            center_img = center_arr[:, :, 0]
            center_b0  = center_arr[:, :, 1]

            binary_map_center = np.where(np.abs(center_b0) < threshold, 0, 1)
            center_b0 = center_b0 * binary_map_center
            center_img = center_img * binary_map_center
            np.divide(center_img, center_b0, out=center_img, where=(center_b0 != 0))

        # Now center_img is normalized; discard the b0 and keep only this
            center_arr = center_img[:, :, np.newaxis]

    # Metadata from center direction
        metadata_map = self.get_metadata(center_idx).detach().cpu().numpy()
        metadata_map = (metadata_map - metadata_map.min()) / (metadata_map.max() - metadata_map.min() + 1e-8)

        center_tensor = torch.tensor(center_arr, dtype=torch.float32).permute(2, 0, 1)
        neighbor_tensor = torch.tensor(neighbor_arr, dtype=torch.float32).permute(2, 0, 1)

        metadata_tensor = torch.tensor(metadata_map[0], dtype=torch.float32).unsqueeze(0)

        center_tensor = center_tensor.unsqueeze(0)
        neighbor_tensor = neighbor_tensor.unsqueeze(0)

        center_slices = []
        for i in range(center_tensor.shape[1]):
            slice_i = center_tensor[0, i].numpy()
            resized = resize(slice_i, self.resize_shape, order=3, mode='reflect', preserve_range=True, anti_aliasing=True)
            center_slices.append(torch.tensor(resized, dtype=torch.float32))
        center_tensor = torch.stack(center_slices)

        neighbor_slices = []
        for i in range(neighbor_tensor.shape[1]):
            slice_i = neighbor_tensor[0, i].numpy()
            resized = resize(slice_i, self.resize_shape, order=3, mode='reflect', preserve_range=True, anti_aliasing=True)
            neighbor_slices.append(torch.tensor(resized, dtype=torch.float32))
        neighbor_tensor = torch.stack(neighbor_slices)

        center_tensor = center_tensor.unsqueeze(0)
        neighbor_tensor = neighbor_tensor.unsqueeze(0)

        center_tensor = center_tensor.squeeze(0)
        neighbor_tensor = neighbor_tensor.squeeze(0)

        neighbor_tensor = torch.cat([neighbor_tensor, metadata_tensor], dim=0)

        return neighbor_tensor, center_tensor