import os
import math
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from models.generator_128 import Unet_Generator 
from models.discriminator_128 import PatchGAN_Discriminator
from models.losses import GANLoss
from models.dataset_multishell import DiffusionDataset

# Data paths.
DATA_PATH = "/data/people/jamesgrist/Desktop/multishell_all_directions"
TRAINING_PATH = os.path.join(DATA_PATH, "TRAINING")
VALIDATION_PATH = os.path.join(DATA_PATH, "VALIDATION")

# Fixed hyperparameters.
input_nc = 5
output_nc = 1
beta1 = 0.5
lambda_L1 = 100  # Weight for L1 loss term
resize_shape = (128, 128)

# Directory for saving checkpoints.
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# CSV file to store trial results.
csv_file = os.path.join(checkpoint_dir, "trial_results.csv")
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "learning_rate", "epochs", "batch_size", "avg_val_loss_G", "trial_time"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global counter for unique trial IDs.
trial_index = 0

def objective(params):
    global trial_index
    trial_id = trial_index
    trial_index += 1

    # Start trial timer.
    trial_start = time.time()
    
    # Extract hyperparameters.
    lr = params['learning_rate']
    num_epochs = int(params['epochs'])
    batch_size = int(params['batch_size'])
    
    tqdm.write(f"Starting Trial {trial_id}: lr={lr:.5f}, epochs={num_epochs}, batch_size={batch_size}")
    
    # Initialize wandb run (disabled mode).
    run = wandb.init(
        entity="simo_projects",
        project="DTI_GAN_Hyperopt",
        config={
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_size
        },
        mode="disabled",
        reinit=True
    )
    
    # Initialize models.
    netG = Unet_Generator(input_nc, output_nc, ngf=64).to(device)
    netD = PatchGAN_Discriminator(input_nc + output_nc, ndf=64, patch_size=70).to(device)
    
    # Loss functions.
    gan_loss_fn = GANLoss('vanilla').to(device)
    l1_loss_fn = nn.L1Loss().to(device)
    
    # Optimizers.
    optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Create datasets and dataloaders.
    train_dataset = DiffusionDataset(TRAINING_PATH, resize_shape=resize_shape)
    val_dataset = DiffusionDataset(VALIDATION_PATH, resize_shape=resize_shape)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Training loop.
    for epoch in range(num_epochs):
        netG.train()
        netD.train()
        
        # Use a simple one-line progress bar.
        train_bar = tqdm(train_dataloader, desc=f"Trial {trial_id} Epoch {epoch+1}/{num_epochs} [Training]", 
                         leave=False, bar_format="{l_bar}{bar:20}{r_bar}")
        for input_img, target_img in train_bar:
            input_img, target_img = input_img.to(device), target_img.to(device)
            
            # Train Discriminator.
            optimizer_D.zero_grad()
            pred_real = netD(torch.cat([input_img, target_img], dim=1))
            loss_D_real = gan_loss_fn(pred_real, True)
            fake_img = netG(input_img)
            pred_fake = netD(torch.cat([input_img, fake_img.detach()], dim=1))
            loss_D_fake = gan_loss_fn(pred_fake, False)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()
            
            # Train Generator.
            optimizer_G.zero_grad()
            pred_fake = netD(torch.cat([input_img, fake_img], dim=1))
            loss_G_GAN = gan_loss_fn(pred_fake, True)
            loss_G_L1 = l1_loss_fn(fake_img, target_img) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()
            
            train_bar.set_postfix({"Loss_G": f"{loss_G.item():.4f}", "Loss_D": f"{loss_D.item():.4f}"})
            run.log({"Generator Training Loss": loss_G.item(), "Discriminator Training Loss": loss_D.item()})
        
        # Validation phase.
        netG.eval()
        netD.eval()
        val_loss_G = 0.0
        num_val_batches = 0
        
        val_bar = tqdm(val_dataloader, desc=f"Trial {trial_id} Epoch {epoch+1}/{num_epochs} [Validation]", 
                       leave=False, bar_format="{l_bar}{bar:20}{r_bar}")
        with torch.no_grad():
            for input_img, target_img in val_bar:
                input_img, target_img = input_img.to(device), target_img.to(device)
                fake_img = netG(input_img)
                pred_fake = netD(torch.cat([input_img, fake_img], dim=1))
                loss_G_GAN = gan_loss_fn(pred_fake, True)
                loss_G_L1 = l1_loss_fn(fake_img, target_img) * lambda_L1
                loss_G = loss_G_GAN + loss_G_L1
                
                val_loss_G += loss_G.item()
                num_val_batches += 1
                val_bar.set_postfix({"Loss_G": f"{loss_G.item():.4f}"})
        
        avg_val_loss_G = val_loss_G / num_val_batches
        # At each epoch, we update the progress line for the trial (if desired you can clear epoch bars).
        tqdm.write(f"Trial {trial_id} Epoch {epoch+1}: Avg Generator Loss: {avg_val_loss_G:.4f}")
    
    # Save checkpoints for this trial with unique filenames.
    netG_path = os.path.join(checkpoint_dir, f'netG_ms_trial_{trial_id}.pth')
    netD_path = os.path.join(checkpoint_dir, f'netD_ms_trial_{trial_id}.pth')
    torch.save(netG.state_dict(), netG_path)
    torch.save(netD.state_dict(), netD_path)
    
    run.finish()
    
    trial_time = time.time() - trial_start
    # Write a clean final summary for the trial.
    summary = (f"Trial {trial_id}: lr={lr:.5f}, epochs={num_epochs}, batch_size={batch_size} -> "
               f"Avg Generator Loss: {avg_val_loss_G:.4f}, Time: {trial_time:.2f} sec")
    tqdm.write(summary)
    run.log({"trial_time": trial_time})
    
    # Append trial results to CSV.
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trial_id, lr, num_epochs, batch_size, avg_val_loss_G, trial_time])
    
    return {'loss': avg_val_loss_G, 'status': STATUS_OK}

# Define hyperparameter search space.
space = {
    'learning_rate': hp.loguniform('learning_rate', math.log(1e-4), math.log(1e-1)),
    'epochs': hp.quniform('epochs', 50, 100, 10),
    'batch_size': hp.quniform('batch_size', 4, 16, 4)
}

trials = Trials()

# Overall timer.
overall_start = time.time()

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials
)

overall_time = time.time() - overall_start
print("Best hyperparameters:", best)
print(f"Total hyperparameter search time: {overall_time:.2f} seconds")

