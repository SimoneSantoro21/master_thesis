import torch
import torch.nn as nn

class UnetDownBlock(nn.Module):
    """Defines the downsampling block of the U-Net."""
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, True))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class UnetUpBlock(nn.Module):
    """Defines the upsampling block of the U-Net."""
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class Unet_Generator(nn.Module):
    """
    U-Net generator for 128x128 images.
    
    This version uses 7 downsampling blocks:
        128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 (bottleneck)
    and 7 upsampling blocks with corresponding skip connections:
        1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128.
    """
    def __init__(self, in_channels, out_channels, ngf=64):
        super().__init__()
        # Encoder: 7 down blocks
        self.down1 = UnetDownBlock(in_channels, ngf, normalize=False)   # 128 -> 64
        self.down2 = UnetDownBlock(ngf, ngf * 2)                            # 64 -> 32
        self.down3 = UnetDownBlock(ngf * 2, ngf * 4)                        # 32 -> 16
        self.down4 = UnetDownBlock(ngf * 4, ngf * 8)                        # 16 -> 8
        self.down5 = UnetDownBlock(ngf * 8, ngf * 8)                        # 8 -> 4
        self.down6 = UnetDownBlock(ngf * 8, ngf * 8)                        # 4 -> 2

        self.down7 = UnetDownBlock(ngf * 8, ngf * 8, normalize=False)       # 2 -> 1 (bottleneck)

        # Decoder: 6 up blocks and a final output layer
        self.up1 = UnetUpBlock(ngf * 8, ngf * 8, dropout=True)              # 1 -> 2
        self.up2 = UnetUpBlock(ngf * 16, ngf * 8, dropout=True)             # 2 -> 4
        self.up3 = UnetUpBlock(ngf * 16, ngf * 8, dropout=True)             # 4 -> 8
        self.up4 = UnetUpBlock(ngf * 16, ngf * 4)                           # 8 -> 16
        self.up5 = UnetUpBlock(ngf * 8, ngf * 2)                            # 16 -> 32
        self.up6 = UnetUpBlock(ngf * 4, ngf)                                # 32 -> 64
        self.up7 = nn.Sequential(                                           # 64 -> 128 (final layer)
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder path
        d1 = self.down1(x)   # 128 -> 64
        d2 = self.down2(d1)  # 64 -> 32
        d3 = self.down3(d2)  # 32 -> 16
        d4 = self.down4(d3)  # 16 -> 8
        d5 = self.down5(d4)  # 8 -> 4
        d6 = self.down6(d5)  # 4 -> 2

        d7 = self.down7(d6)  # 2 -> 1 (bottleneck)

        # Decoder path with skip connections
        u1 = self.up1(d7)                          # 1 -> 2
        u2 = self.up2(torch.cat([u1, d6], dim=1))    # 2 -> 4
        u3 = self.up3(torch.cat([u2, d5], dim=1))    # 4 -> 8
        u4 = self.up4(torch.cat([u3, d4], dim=1))    # 8 -> 16
        u5 = self.up5(torch.cat([u4, d3], dim=1))    # 16 -> 32
        u6 = self.up6(torch.cat([u5, d2], dim=1))    # 32 -> 64
        output = self.up7(torch.cat([u6, d1], dim=1)) # 64 -> 128
        
        return output
