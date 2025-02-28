import torch
import torch.nn as nn


class PatchGAN_Discriminator(nn.Module):
    """
    Implementation of PatchGAN discriminator.

    The user can chose between a 70x70 or a 16x16  PatchGAN.
    """
    def __init__(self, input_nc, ndf=64, patch_size=70):
        """
        Initializes the PatchGAN discriminator.
        
        Args:
            input_nc (int):     Number of channels in the input image.
            ndf (int):          Base number of filters.
            patch_size (int):   Determines the receptive field of the discriminator.
                                Choose 70 for a 70x70 PatchGAN (default) or 16 for a 16x16 patch discriminator.
        """
        super().__init__()
        
        if patch_size == 70:
            self.model = nn.Sequential(
                # Layer 1: Convolution with stride 2; no normalization
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
                
                # Layer 2 
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, True),
                
                # Layer 3
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, True),
                
                # Layer 4: Uses stride 1
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, True),
                
                # Output layer: Produces a one-channel output map.
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
            )

        elif patch_size == 16:
            # A smaller discriminator architecture for 16x16 patches.
            self.model = nn.Sequential(
                # Layer 1
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
                
                # Layer 2.
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, True),
                
                # Output layer
                nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=1, padding=1)
            )

        else:
            raise ValueError("Unsupported patch size. Please choose patch_size=70 or patch_size=16.")

    def forward(self, x):
        return self.model(x)
