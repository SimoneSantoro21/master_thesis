import os
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from models.generator_128 import Unet_Generator
from models.dataset import DiffusionDataset

# Constants
BASE_DATA_PATH = "/data/people/jamesgrist/Desktop/DTI_single_direction"
CHECKPOINTS_DIR = "checkpoints"
OUTPUT_BASE = "/data/people/jamesgrist/Desktop/output_patient_{patient_id}_test"
RESIZE_SHAPE = (128, 128)
INPUT_NC = 4
OUTPUT_NC = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference_for_patient(patient_id):
    output_dir = OUTPUT_BASE.format(patient_id=patient_id)
    os.makedirs(output_dir, exist_ok=True)

    for center_index in range(1, 31):
        print(f"\nProcessing center index {center_index} for patient {patient_id}")

        dataset_path = os.path.join(BASE_DATA_PATH, f"dataset_center{center_index}_BET", "TEST")
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, "netG_bvec_3.pth")

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
        if not os.path.isdir(dataset_path):
            print(f"Dataset folder not found: {dataset_path}")
            continue

        netG = Unet_Generator(INPUT_NC, OUTPUT_NC, ngf=64).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        netG.load_state_dict(checkpoint)
        netG.eval()

        dataset = DiffusionDataset(dataset_path, center_index=center_index, resize_shape=RESIZE_SHAPE)

        slice_indices = [i for (pid, i) in dataset.keys if pid == patient_id]
        if not slice_indices:
            print(f"No slices found for patient {patient_id} in center {center_index}")
            continue

        for slice_idx in tqdm(slice_indices, desc=f"Inference center {center_index}"):
            try:
                input_tensor, _ = dataset[(patient_id, slice_idx)]
                input_tensor = input_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    output_tensor = netG(input_tensor)

                output_np = output_tensor.squeeze().cpu().numpy()

                # Normalize from [-1, 1] to [0, 1] if needed
                if output_np.min() >= -1 and output_np.max() <= 1:
                    output_np = (output_np + 1) / 2

                output_img = nib.Nifti1Image(output_np, affine=np.eye(4))
                out_path = os.path.join(output_dir, f"output_{patient_id}_center{center_index}_slice{slice_idx}.nii.gz")
                nib.save(output_img, out_path)

            except Exception as e:
                print(f"Failed processing slice {slice_idx} for center {center_index}: {e}")

if __name__ == "__main__":
    patient_id = "003"
    run_inference_for_patient(patient_id)
