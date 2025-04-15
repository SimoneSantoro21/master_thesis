import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.generator_128 import Unet_Generator
from models.discriminator_128 import PatchGAN_Discriminator
from models.losses import GANLoss
from models.dataset import DiffusionDataset

import wandb


num_epochs = 70
batch_size = 12
input_nc = 4
output_nc = 1
lr = 0.0007851029472710995
beta1 = 0.5
lambda_L1 = 100  # Weight for L1 loss term


DATA_PATH = "/data/people/jamesgrist/Desktop/"

for bvec in range(3, 4):
    DATASET = os.path.join(DATA_PATH, f'dataset_center{bvec}_BET')
    TRAINING_PATH = os.path.join(DATASET, "TRAINING")
    VALIDATION_PATH = os.path.join(DATASET, "VALIDATION")

    # Directory for saving checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Start a new wandb run to track this script.
    run = wandb.init(
        entity="simo_projects",
        project="DTI_GAN",
        config={
            "learning_rate": lr,
            "architecture": "CNN",
            "dataset": "",
            "epochs": num_epochs,
            "batch_size": batch_size
        },
        mode="disabled",
        reinit=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Generator and Discriminator
    netG = Unet_Generator(input_nc, output_nc, ngf=64).to(device)
    netD = PatchGAN_Discriminator(input_nc + output_nc, ndf=64, patch_size=70).to(device)

    # Loss Functions and Optimizers
    gan_loss_fn = GANLoss('vanilla').to(device)  # Or use 'lsgan'
    l1_loss_fn = nn.L1Loss().to(device)

    optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    # Data Loaders
    train_dataset = DiffusionDataset(TRAINING_PATH, center_index=bvec, resize_shape=(128, 128))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = DiffusionDataset(VALIDATION_PATH, center_index=bvec, resize_shape=(128, 128))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("Starting training...")

    for epoch in range(num_epochs):
        netG.train()
        netD.train()

        # Create a tqdm progress bar for the training loop.
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
        for i, (input_img, target_img) in enumerate(train_bar):
            input_img, target_img = input_img.to(device), target_img.to(device)

            ### Train Discriminator ###
            optimizer_D.zero_grad()
            # Real pair: input + target
            pred_real = netD(torch.cat([input_img, target_img], dim=1))
            loss_D_real = gan_loss_fn(pred_real, True)
            # Fake pair: input + generated (detach generator)
            fake_img = netG(input_img)
            pred_fake = netD(torch.cat([input_img, fake_img.detach()], dim=1))
            loss_D_fake = gan_loss_fn(pred_fake, False)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()


            ### Train Generator ###
            optimizer_G.zero_grad()
            # Discriminator on fake pair (input + generated)
            pred_fake = netD(torch.cat([input_img, fake_img], dim=1))
            loss_G_GAN = gan_loss_fn(pred_fake, True)
            loss_G_L1 = l1_loss_fn(fake_img, target_img) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()


            # Update tqdm description with current losses
            train_bar.set_postfix({
                "Loss_G": loss_G.item(),
                "Loss_D": loss_D.item()
            })

            run.log({"Generator Training Loss": loss_G, "Discriminator Training Loss": loss_D})

        # Validation Phase
        netG.eval()
        netD.eval()
        val_loss_G = 0.0
        val_loss_D = 0.0
        num_val_batches = 0

        # Disable gradients for validation.
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)
            for input_img, target_img in val_bar:
                input_img, target_img = input_img.to(device), target_img.to(device)

                # Generator forward pass
                fake_img = netG(input_img)
                pred_fake = netD(torch.cat([input_img, fake_img], dim=1))
                loss_G_GAN = gan_loss_fn(pred_fake, True)
                loss_G_L1 = l1_loss_fn(fake_img, target_img) * lambda_L1
                loss_G = loss_G_GAN + loss_G_L1

                # Discriminator on real and fake pairs
                pred_real = netD(torch.cat([input_img, target_img], dim=1))
                loss_D_real = gan_loss_fn(pred_real, True)
                pred_fake = netD(torch.cat([input_img, fake_img], dim=1))
                loss_D_fake = gan_loss_fn(pred_fake, False)
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                val_loss_G += loss_G.item()
                val_loss_D += loss_D.item()
                num_val_batches += 1

                val_bar.set_postfix({
                    "Loss_G": loss_G.item(),
                    "Loss_D": loss_D.item()
                })

        avg_val_loss_G = val_loss_G / num_val_batches
        avg_val_loss_D = val_loss_D / num_val_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation: Avg Loss_G: {avg_val_loss_G:.4f}, Avg Loss_D: {avg_val_loss_D:.4f}")

    # Save the generator and discriminator models every 10 epochs.
    torch.save(netG.state_dict(), os.path.join(checkpoint_dir, f'netG_bvec_{bvec}.pth'))
    torch.save(netD.state_dict(), os.path.join(checkpoint_dir, f'netD_bvec_{bvec}.pth'))

    run.finish()
    print("Training complete.")

