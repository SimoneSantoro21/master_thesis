import torch
import torch.nn as nn

class UnetDownBlock(nn.Module):
    """ Defines the downsampling block of the U-Net """

    def __init__(self, in_channels, out_channels, normalize=True):
        """
        Initialize the class.

        Parameters:
            in_channels (int):  number of channels in the input images
            out_channels (int): number of channels in the output images
            normalize (bool):   option to turn on normalization layers
        """
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUpBlock(nn.Module):
    """ Defines the upsampling block of the U-Net """

    def __init__(self, in_channels, out_channels, dropout=False):
        """
        Initialize the class.

        Parameters:
            in_channels (int):  number of channels in the input images
            out_channels (int): number of channels in the output images
            dropout (bool):     option to turn on dropout layers
        """
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
    Implementation of Unet Generator module.
        
    It assumes 8 downsampling steps (producing features at 256->128->64->32->16->8->4->2->1 resolution)
    and then an 8-step upsampling path with skip connections.    
    """
    def __init__(self, in_channels, out_channels, ngf=64):
        """
        Construct the U-Net Generator.

        Parameters:
            in_channels (int):  number of channels in the input images
            out_channels (int): number of channels in the output images
            ngf (int):          number of filters in the firs convolution layer

        The generator is built by individually defining each block
        """
        super().__init__()
        
        # Encoder (Downsampling) layers
        # The first layer does not use normalization.
        self.down1 = UnetDownBlock(in_channels, ngf, normalize=False)     
        self.down2 = UnetDownBlock(ngf, ngf * 2)                         
        self.down3 = UnetDownBlock(ngf * 2, ngf * 4)                     
        self.down4 = UnetDownBlock(ngf * 4, ngf * 8)                     
        self.down5 = UnetDownBlock(ngf * 8, ngf * 8)                     
        self.down6 = UnetDownBlock(ngf * 8, ngf * 8)                     
        self.down7 = UnetDownBlock(ngf * 8, ngf * 8)   

        # The innermost layer (bottleneck) – no normalization here
        self.down8 = UnetDownBlock(ngf * 8, ngf * 8, normalize=False)    

        # Decoder (Upsampling) layers with skip connections.
        # After each upsampling, the result is concatenated with the corresponding encoder feature map.
        self.up1 = UnetUpBlock(ngf * 8, ngf * 8, dropout=True)           
        self.up2 = UnetUpBlock(ngf * 16, ngf * 8, dropout=True)          
        self.up3 = UnetUpBlock(ngf * 16, ngf * 8, dropout=True)          
        self.up4 = UnetUpBlock(ngf * 16, ngf * 8)                        
        self.up5 = UnetUpBlock(ngf * 16, ngf * 4)                        
        self.up6 = UnetUpBlock(ngf * 8, ngf * 2)                         
        self.up7 = UnetUpBlock(ngf * 4, ngf)                             

        # Outermost upsampling: maps to the desired output channels and applies Tanh
        self.up8 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )                                                              

    def forward(self, x):
        # Encoder path (store outputs for skip connections)
        d1 = self.down1(x)   # 256 -> 128
        d2 = self.down2(d1)  # 128 -> 64
        d3 = self.down3(d2)  # 64 -> 32
        d4 = self.down4(d3)  # 32 -> 16
        d5 = self.down5(d4)  # 16 -> 8
        d6 = self.down6(d5)  # 8 -> 4
        d7 = self.down7(d6)  # 4 -> 2

        # Bottleneck
        d8 = self.down8(d7)  # 2 -> 1 

        # Decoder path with skip connections (concatenation along the channel dimension)
        u1 = self.up1(d8)                # 1 -> 2
        u1 = torch.cat([u1, d7], dim=1)    # Concatenate with d7
        u2 = self.up2(u1)                # 2 -> 4
        u2 = torch.cat([u2, d6], dim=1)    # Concatenate with d6
        u3 = self.up3(u2)                # 4 -> 8
        u3 = torch.cat([u3, d5], dim=1)    # Concatenate with d5
        u4 = self.up4(u3)                # 8 -> 16
        u4 = torch.cat([u4, d4], dim=1)    # Concatenate with d4
        u5 = self.up5(u4)                # 16 -> 32
        u5 = torch.cat([u5, d3], dim=1)    # Concatenate with d3
        u6 = self.up6(u5)                # 32 -> 64
        u6 = torch.cat([u6, d2], dim=1)    # Concatenate with d2
        u7 = self.up7(u6)                # 64 -> 128
        u7 = torch.cat([u7, d1], dim=1)    # Concatenate with d1

        output = self.up8(u7)            # 128 -> 256
        return output