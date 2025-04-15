import torch
import torch.nn as nn

class PatchGAN_Discriminator(nn.Module):
    """
    PatchGAN discriminator modified for 128x128 images.
    
    The discriminator supports two modes:
      - patch_size=70: A 70×70 PatchGAN variant.
      - patch_size=16: A 16×16 PatchGAN variant.
    """
    def __init__(self, input_nc, ndf=64, patch_size=70):
        """
        Initializes the PatchGAN discriminator.
        
        Args:
            input_nc (int): Number of channels in the input image.
            ndf (int): Base number of filters.
            patch_size (int): Determines the receptive field of the discriminator.
                              Use 70 for a 70×70 PatchGAN (default) or 16 for a 16×16 patch discriminator.
        """
        super().__init__()
        
        if patch_size == 70:
            # For 128x128 images, we use three downsampling layers with stride 2 instead of four.
            self.model = nn.Sequential(
                # Layer 1: 128 -> 64 (no normalization)
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
                
                # Layer 2: 64 -> 32
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, True),
                
                # Layer 3: 32 -> 16
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, True),
                
                # Layer 4: Convolution with stride 1 to expand the receptive field.
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, True),
                
                # Output layer: Produces a one-channel output map.
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
            )
            
        elif patch_size == 16:
            # For the 16x16 variant, we add one more downsampling block.
            self.model = nn.Sequential(
                # Layer 1: 128 -> 64
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
                
                # Layer 2: 64 -> 32
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, True),
                
                # Layer 3: 32 -> 16
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, True),
                
                # Output layer: Convolution with stride 1.
                nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=1)
            )
        else:
            raise ValueError("Unsupported patch size. Please choose patch_size=70 or patch_size=16.")

    def forward(self, x):
        return self.model(x)
