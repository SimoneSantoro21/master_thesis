import torch
import torch.nn as nn
import pytest
from models.losses import GANLoss

# --- Tests for get_target_tensor ---

def test_get_target_tensor_shape_real():
    """
    Test get_target_tensor for real images.

    GIVEN:  A torch.Tensor prediction of shape (batch_size, channels, height, width)
    WHEN:   get_target_tensor is called with target_is_real=True
    THEN:   The output tensor has the same shape as prediction and is filled with the real label (1.0)
    """
    prediction = torch.randn(2, 3, 4, 4)
    loss_fn = GANLoss(gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0)
    target_tensor = loss_fn.get_target_tensor(prediction, target_is_real=True)

    assert target_tensor.shape == prediction.shape
    assert torch.all(target_tensor == 1.0)


def test_get_target_tensor_shape_fake():
    """
    Test get_target_tensor for fake images.

    GIVEN:  A torch.Tensor prediction of shape (batch_size, channels, height, width)
    WHEN:   get_target_tensor is called with target_is_real=False
    THEN:   The output tensor has the same shape as prediction and is filled with the fake label (0.0)
    """
    prediction = torch.randn(2, 3, 4, 4)
    loss_fn = GANLoss(gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0)
    target_tensor = loss_fn.get_target_tensor(prediction, target_is_real=False)

    assert target_tensor.shape == prediction.shape
    assert torch.all(target_tensor == 0.0)


# --- Tests for GANLoss forward method ---

def test_forward_loss_lsgan_real():
    """
    Test GANLoss forward method in lsgan mode for real images.

    GIVEN:  A prediction tensor filled with 1.0 (real label value)
    WHEN:   Forwarded through GANLoss with target_is_real=True
    THEN:   The MSE loss should be zero
    """
    loss_fn = GANLoss(gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0)
    prediction = torch.ones(2, 3, 4, 4)
    loss = loss_fn(prediction, target_is_real=True)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_forward_loss_lsgan_fake():
    """
    Test GANLoss forward method in lsgan mode for fake images.

    GIVEN:  A prediction tensor filled with 0.0 (fake label value)
    WHEN:   Forwarded through GANLoss with target_is_real=False
    THEN:   The MSE loss should be zero
    """
    loss_fn = GANLoss(gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0)
    prediction = torch.zeros(2, 3, 4, 4)
    loss = loss_fn(prediction, target_is_real=False)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_forward_loss_vanilla_real():
    """
    Test GANLoss forward method in vanilla mode for real images.

    GIVEN:  A prediction tensor with high positive values
    WHEN:   Forwarded through GANLoss with target_is_real=True
    THEN:   The BCEWithLogitsLoss should be nearly zero
    """
    loss_fn = GANLoss(gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0)
    prediction = torch.full((2, 3, 4, 4), 100.0)
    loss = loss_fn(prediction, target_is_real=True)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_forward_loss_vanilla_fake():
    """
    Test GANLoss forward method in vanilla mode for fake images.

    GIVEN:  A prediction tensor with low negative values
    WHEN:   Forwarded through GANLoss with target_is_real=False
    THEN:   The BCEWithLogitsLoss should be nearly zero
    """
    loss_fn = GANLoss(gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0)
    prediction = torch.full((2, 3, 4, 4), -100.0)
    loss = loss_fn(prediction, target_is_real=False)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_invalid_gan_mode():
    """
    Test that initializing GANLoss with an unsupported gan_mode raises a NotImplementedError.

    GIVEN:  An invalid gan_mode string
    WHEN:   Instantiating GANLoss
    THEN:   A NotImplementedError should be raised
    """
    with pytest.raises(NotImplementedError) as excinfo:
        _ = GANLoss(gan_mode='unsupported')
    
    assert 'gan mode' in str(excinfo.value)
