import torch
import torch.nn as nn
from models.generator import UnetDownBlock, UnetUpBlock, Unet_Generator


# --- Tests for UnetDownBlock ---

def test_unet_down_block_output_shape_with_normalization():
    """
    Test UnetDownBlock whith normalization enabled.

    GIVEN:  A torch.tensor of shape (batch_size, in_channels, height, width)
    WHEN:   UnetDwonBlock is applied
    THEN:   The output is a tensor (batch_size, out_channels, height // 2, width // 2)
    """
    in_channels = 3
    out_channels = 64
    block = UnetDownBlock(in_channels, out_channels, normalize=True)

    x = torch.randn(1, in_channels, 256, 256).float()
    y = block(x)

    assert y.shape == (1, out_channels, 128, 128)


def test_unet_down_block_without_normalization():
    """
    Testing the UnetDownBlock with normalization not enabled.

    GIVEN:  An instance of UnetDownBlock with parameter normalize=False
    THEN:   No BatchNorm2d layer is present in the block 
    """
    in_channels = 3
    out_channels = 64
    block = UnetDownBlock(in_channels, out_channels, normalize=False)

    for module in block.model:
        assert not isinstance(module, nn.BatchNorm2d)



# --- Tests for UnetUpBlock ---

def test_unet_up_block_output_shape_with_dropout():
    """
    Testing UnetUpBlock with dropout enabled.

    GIVEN:  A torch.tensor of shape (batch_size, in_channels, height, width)
    WHEN:   UnetUpBlock is applied
    THEN:   The output is a tensor (batch_size, out_channels, height * 2, width * 2)
    """
    in_channels = 128
    out_channels = 64
    block = UnetUpBlock(in_channels, out_channels, dropout=True)

    x = torch.randn(1, in_channels, 2, 2)
    y = block(x)

    assert y.shape == (1, out_channels, 4, 4)


def test_unet_up_block_without_dropout():
    """
    Testing the UnetUpBlock with dropout not enabled.

    GIVEN:  An instance of UnetUpBlock with parameter dropout=False
    THEN:   No Dropout2d layer is present in the block 
    """
    in_channels = 128
    out_channels = 64
    block = UnetUpBlock(in_channels, out_channels, dropout=True)
    # Check that a dropout layer exists in the sequential model.
    for module in block.model:
        assert not isinstance(module, nn.Dropout2d)


# --- Tests for Unet_Generator ---

def test_unet_generator_output_shape():
    """
    Test UnetGenerator output shape.

    GIVEN:  A torch.tensor of shape (batch_size, in_channels, height, width)
    WHEN:   Forwarded in the generator
    THEN:   The output is a tensor (batch_size, out_channels, height, width)
    """
    in_channels = 3
    out_channels = 3
    ngf = 64
    generator = Unet_Generator(in_channels, out_channels, ngf=ngf)

    x = torch.randn(1, in_channels, 256, 256)
    y = generator(x)

    assert y.shape == (1, out_channels, 256, 256)


def test_unet_generator_output_range():
    """
    Test UnetGenerator output range.
    Final activation function is Tanh() -> ouput range should be [-1, 1]

    GIVEN:  A torch.tensor of shape (batch_size, in_channels, height, width)
    WHEN:   Forwarded in the generator
    THEN:   The output is a tensor with values in the range [-1, 1]
    """
    in_channels = 3
    out_channels = 3
    generator = Unet_Generator(in_channels, out_channels)
    x = torch.randn(1, in_channels, 256, 256)
    y = generator(x)

    assert y.min() >= -1.0 and y.max() <= 1.0


def test_unet_generator_backward():
    """
    Test UnetGenerator backpropagation with a dummy loss function (mean).

    GIVEN:  The output of the generator
    WHEN:   Loss is computed and backpropagated
    THEN:   The gradients must be not None
    """
    in_channels = 3
    out_channels = 3
    generator = Unet_Generator(in_channels, out_channels)
    x = torch.randn(1, in_channels, 256, 256, requires_grad=True)
    y = generator(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None