"""Loss function tests"""

import pytest
import torch

from compyute.nn import BCELoss, CrossEntropyLoss, DiceLoss, MSELoss
from tests.utils import get_random_floats, get_random_integers, is_close

testdata = [(8, 16), (8, 16, 32), (8, 16, 32, 64)]
dice_data = [(4, 8, 16, 16), (8, 16, 32, 32)]


@pytest.mark.parametrize("shape", testdata)
def test_mse_loss(shape) -> None:
    """Test for the mean squared error loss."""

    # init compyute loss
    compyute_loss = MSELoss()

    # init torch loss
    torch_loss = torch.nn.MSELoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    compyute_t, torch_t = get_random_floats(shape)
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x, torch_t)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", testdata)
def test_cross_entropy_loss(shape) -> None:
    """Test for the cross entropy loss."""

    # init compyute loss
    compyute_loss = CrossEntropyLoss()

    # init torch loss
    torch_loss = torch.nn.CrossEntropyLoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    torch_x_ma = torch.moveaxis(torch_x, -1, 1) if len(shape) > 2 else torch_x
    compyute_t, torch_t = get_random_integers(shape[:-1], high=shape[-1])
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x_ma, torch_t)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", testdata)
def test_bce_loss(shape) -> None:
    """Test for the binary cross entropy loss."""

    # init compyute loss
    compyute_loss = BCELoss()

    # init torch loss
    torch_loss = torch.nn.BCEWithLogitsLoss()

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    compyute_t, torch_t = get_random_floats(shape)
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_loss(torch_x, torch_t)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_close(compyute_dx, torch_x.grad)


@pytest.mark.parametrize("shape", dice_data)
def test_dice_loss(shape) -> None:
    """Test for the dice loss."""

    # init compyute loss
    compyute_loss = DiceLoss()

    # init torch loss https://github.com/shuaizzZ/Dice-Loss-PyTorch/tree/master
    def torch_dice_loss(x, t):
        eps = 1e-5
        N, C = x.size()[:2]
        predict = x.view(N, C, -1)
        target = t.view(N, 1, -1)

        predict = torch.nn.functional.softmax(predict, dim=1)
        target_onehot = torch.zeros(predict.size())
        target_onehot.scatter_(1, target, 1)

        intersection = torch.sum(predict * target_onehot, dim=2)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)
        dice_coef = (2 * intersection + eps) / (union + eps)
        return 1 - torch.mean(dice_coef)

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    compyute_t, torch_t = get_random_integers((shape[0], *shape[2:]), high=shape[1])
    compyute_y = compyute_loss(compyute_x, compyute_t)
    torch_y = torch_dice_loss(torch_x, torch_t)
    assert is_close(compyute_y, torch_y, tol=1e-4)

    # backward
    compyute_dx = compyute_loss.backward()
    torch_y.backward()
    assert is_close(compyute_dx, torch_x.grad)
