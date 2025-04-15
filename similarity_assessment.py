import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

from models.generator_128 import Unet_Generator
from models.dataset import DiffusionDataset

# Configuration
center_index = 10        
resize_shape = (128, 128)     
input_nc = 4                  # Number of input channels for the model.
output_nc = 1                 # Number of output channels (ground truth).
TEST_PATH = f"/data/people/jamesgrist/Desktop/DTI_single_direction/dataset_center{center_index}_BET/TEST"
checkpoint_file = "checkpoints/netG_bvec_3.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
netG = Unet_Generator(input_nc, output_nc, ngf=64).to(device)
checkpoint = torch.load(checkpoint_file, map_location=device)
netG.load_state_dict(checkpoint)
netG.eval()

# Create Dataset
test_dataset = DiffusionDataset(TEST_PATH, center_index=center_index, resize_shape=resize_shape)

# Metrics Arrays
psnr_list = []
ssim_list = []

# Iterate over all samples in the dataset.
for idx in range(len(test_dataset)):
    neighbor_tensor, center_tensor = test_dataset[idx]
    
    # Add a batch dimension so that shape becomes [1, C, H, W] before feeding to the model.
    input_tensor = neighbor_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = netG(input_tensor)
    
    # Remove the batch dimension and convert tensors to NumPy arrays.
    output_img = output_tensor.squeeze(0).cpu().numpy()
    gt_img = center_tensor.cpu().numpy()
    
    # If the image tensor has a single channel (shape: [1, H, W]), squeeze that channel.
    if output_img.shape[0] == 1:
        output_img = output_img[0]
    if gt_img.shape[0] == 1:
        gt_img = gt_img[0]
    
    # Compute PSNR.
    # Since the images are normalized to [0,1], data_range is set to 1.
    psnr_val = peak_signal_noise_ratio(gt_img, output_img, data_range=1)
    psnr_list.append(psnr_val)
    
    # Compute SSIM.
    ssim_val = structural_similarity(gt_img, output_img, data_range=1)
    ssim_list.append(ssim_val)
    
    print(f"Sample {idx}: PSNR = {psnr_val:.2f}, SSIM = {ssim_val:.4f}")

# Convert lists to numpy arrays to compute statistics.
psnr_array = np.array(psnr_list)
ssim_array = np.array(ssim_list)

avg_psnr = np.mean(psnr_array)
std_psnr = np.std(psnr_array)
avg_ssim = np.mean(ssim_array)
std_ssim = np.std(ssim_array)

print("\nOverall Metrics:")
print(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")