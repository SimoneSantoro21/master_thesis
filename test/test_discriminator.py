import torch
import pytest
import torch.nn as nn
from models.discriminator import PatchGAN_Discriminator

def test_patchgan_discriminator_70_output_shape():
    """
    Test the PatchGAN_Discriminator using patch_size=70.
    
    GIVEN:  A 256x256 input image 
    WHEN:   Is passed through the 70x70 PatchGAN discriminator
    THEN:   The output shape is as expected (batch_size, 1, 30, 30)
    """
    input_nc = 3
    ndf = 64
    patch_size = 70

    discriminator = PatchGAN_Discriminator(input_nc, ndf, patch_size=patch_size)

    x = torch.randn(1, input_nc, 256, 256)
    y = discriminator(x)

    assert y.shape == (1, 1, 30, 30)


def test_patchgan_discriminator_16_output_shape():
    """
    Test the PatchGAN_Discriminator using patch_size=16.
    
    GIVEN:  A 256x256 input image 
    WHEN:   Is passed through the 16x16 PatchGAN discriminator
    THEN:   The output shape is as expected (batch_size, 1, 63, 63)
    """
    input_nc = 3
    ndf = 64
    patch_size = 16

    discriminator = PatchGAN_Discriminator(input_nc, ndf, patch_size=patch_size)

    x = torch.randn(1, input_nc, 256, 256)
    y = discriminator(x)

    assert y.shape == (1, 1, 63, 63)


def test_patchgan_discriminator_invalid_patch_size():
    """
    Test that PatchGAN_Discriminator raises a ValueError for unsupported patch sizes.
    
    GIVEN:  An invalid pstch_size
    WHEN:   Initializing the discriminator
    THEN:   An "Unsupported patch size" error is raised
    """
    input_nc = 3
    ndf = 64
    patch_size = 50  # Unsupported patch size.

    with pytest.raises(ValueError) as excinfo:
        _ = PatchGAN_Discriminator(input_nc, ndf, patch_size=patch_size)

    assert "Unsupported patch size" in str(excinfo.value)


def test_patchgan_discriminator_backward():
    """
    Test PatchGAN_Discriminator backpropagation with a dummy loss function (mean).

    GIVEN:  The output of the discriminator
    WHEN:   Dummy Loss is computed and backpropagated
    THEN:   The gradients must be not None
    """
    input_nc = 3
    ndf = 64
    patch_size = 70
    # Initialize the discriminator with patch_size=70.
    discriminator = PatchGAN_Discriminator(input_nc, ndf, patch_size=patch_size)
    # Create a dummy input tensor with requires_grad=True to track gradients.
    x = torch.randn(1, input_nc, 256, 256, requires_grad=True)
    # Forward pass: obtain the discriminator's output.
    y = discriminator(x)
    # Define a simple loss as the mean of the output.
    loss = y.mean()
    # Backward pass: compute gradients.
    loss.backward()
    # Verify that gradients have been computed for the input tensor.
    assert x.grad is not None