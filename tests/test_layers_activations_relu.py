"""ReLu activation layer tests"""

import numpy as np
import torch
import torch.nn.functional as F
import walnut


def test_relu_y_cpu() -> None:
    walnut_x = walnut.randn((10, 10, 10))
    torch_x = torch.from_numpy(walnut_x.data)
    torch_x.requires_grad = True

    relu = walnut.nn.layers.ReLU()
    relu.training = True

    walnut_y = relu(walnut_x).data
    torch_y = F.relu(torch_x).detach().numpy()

    assert np.allclose(walnut_y, torch_y)


def test_relu_y_cuda() -> None:
    if not walnut.cuda.is_available():
        pass

    walnut_x = walnut.randn((10, 10, 10))
    torch_x = torch.from_numpy(walnut_x.data)
    torch_x.requires_grad = True
    walnut_x.to_device("cuda")

    relu = walnut.nn.layers.ReLU()
    relu.training = True

    walnut_y = walnut.cuda.cupy_to_numpy(relu(walnut_x).data)
    torch_y = F.relu(torch_x).detach().numpy()

    assert np.allclose(walnut_y, torch_y)


def test_relu_dx_cpu() -> None:
    walnut_x = walnut.randn((10, 10, 10))
    torch_x = torch.from_numpy(walnut_x.data)
    torch_x.requires_grad = True

    relu = walnut.nn.layers.ReLU()
    relu.training = True

    _ = relu(walnut_x)
    torch_y = F.relu(torch_x)

    walnut_dy = walnut.randn(walnut_x.shape)
    torch_dy = torch.from_numpy(walnut_dy.data)

    walnut_dx = relu.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    assert np.allclose(walnut_dx, torch_x.grad.numpy())


def test_relu_dx_cuda() -> None:
    if not walnut.cuda.is_available():
        pass

    walnut_x = walnut.randn((10, 10, 10))
    torch_x = torch.from_numpy(walnut_x.data)
    torch_x.requires_grad = True
    walnut_x.to_device("cuda")

    relu = walnut.nn.layers.ReLU()
    relu.training = True

    _ = relu(walnut_x)
    torch_y = F.relu(torch_x)

    walnut_dy = walnut.randn(walnut_x.shape)
    torch_dy = torch.from_numpy(walnut_dy.data)
    walnut_dy.to_device("cuda")

    walnut_dx = walnut.cuda.cupy_to_numpy(relu.backward(walnut_dy.data))
    torch_y.backward(torch_dy)

    assert np.allclose(walnut_dx, torch_x.grad.numpy())
