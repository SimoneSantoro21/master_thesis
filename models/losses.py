import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """
    Implementation of GAN Loss class.
    
    It supports both vanilla GAN loss (using BCEWithLogitsLoss) and LSGAN (using MSELoss), as implemented in the
    original pix2pix repository.
    """
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """
        Initializing GANLoss.

        Parameters:
            gan_mode (str): Type of GAN objective ('vanilla' or 'lsgan').
            target_real_label (float): Label for real images.
            target_fake_label (float): Label for fake images.
        """
        super(GANLoss, self).__init__()
        # Register buffers so that these tensors are part of the state but not learnable.
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode {} not implemented'.format(gan_mode))
    
    def get_target_tensor(self, prediction, target_is_real):
        """
        Create label tensors with the same size as the prediction.
        
        Parameters:
            prediction (torch.Tensor): The output from the discriminator.
            target_is_real (bool): Whether the ground truth label is for real images.
            
        Returns:
            torch.Tensor: Target tensor filled with either real or fake label.
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def forward(self, prediction, target_is_real):
        """
        Calculate loss given discriminator's output and ground truth labels.
        
        Parameters:
            prediction (torch.Tensor): The discriminator output.
            target_is_real (bool): True if ground truth label is for real images.
            
        Returns:
            torch.Tensor: The calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss