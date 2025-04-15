import os
import torch
import matplotlib.pyplot as plt
from models.generator_128 import Unet_Generator 
from models.dataset import DiffusionDataset

# Configuration.
center_index = 3

DATA_PATH = f"/data/people/jamesgrist/Desktop/DTI_single_direction/dataset_center{center_index}_BET"
TEST_PATH = os.path.join(DATA_PATH, "TEST")
resize_shape = (128, 128)
input_nc = 4
output_nc = 1

# Path to the generator checkpoint (update as needed).
checkpoint_file = f"checkpoints/netG_bvec_{center_index}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_sample(patient_id, slice_idx):
    # Initialize the generator and load its checkpoint.
    netG = Unet_Generator(input_nc, output_nc, ngf=64).to(device)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    netG.load_state_dict(checkpoint)
    netG.eval()

    # Create the validation dataset.
    val_dataset = DiffusionDataset(TEST_PATH, center_index=center_index, resize_shape=resize_shape)

    # Retrieve the sample using tuple indexing.
    try:
        # __getitem__ returns (neighbor_tensor, center_tensor)
        input_tensor, gt_tensor = val_dataset[(patient_id, slice_idx)]
    except Exception as e:
        print(f"Error retrieving sample for patient {patient_id} slice {slice_idx}: {e}")
        return

    # Add a batch dimension for inference.
    input_tensor = input_tensor.unsqueeze(0).to(device)
    print(input_tensor.shape)

    # Run the generator.
    with torch.no_grad():
        output_tensor = netG(input_tensor)

    # Remove batch dimension and convert tensors to NumPy arrays.
    output_img = output_tensor.squeeze().cpu().numpy()
    gt_img = gt_tensor.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Main title for the entire figure
    fig.suptitle(f"Patient {patient_id} Slice {slice_idx}", fontsize=20)

    # First subplot: Ground Truth
    axs[0].imshow(gt_img, cmap='gray')
    axs[0].set_title("Ground Truth", fontsize=16)
    axs[0].axis('off')

    # Second subplot: Model Output
    axs[1].imshow(output_img, cmap='gray')
    axs[1].set_title("Model Output", fontsize=16)
    axs[1].axis('off')

    # Adjust layout to fit the title and subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()

if __name__ == "__main__":
    # Prompt user for patient ID and slice index.
    test_dataset = DiffusionDataset(TEST_PATH, center_index=center_index, resize_shape=resize_shape)
    patient_id = "003"
    slice_idx = 38
    visualize_sample(patient_id, slice_idx)
