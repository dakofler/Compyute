"""Recurrent layer tests"""

import torch
import walnut
from tests.test_utils import get_vals, get_params, validate


B, Cin, Ch, X = (10, 10, 10, 10)


# Recurrent
def test_recurrent_cpu() -> None:
    results = []
    shape_x = (B, X, Cin)
    shape_w_in = (Cin, Ch)
    shape_b_in = (Ch,)
    shape_w_h = (Ch, Ch)
    shape_b_h = (Ch,)

    # forward
    walnut_x, torch_x = get_vals(shape_x)
    walnut_w_in, torch_w_in = get_params(shape_w_in, T=True)
    walnut_b_in, torch_b_in = get_params(shape_b_in)
    walnut_w_h, torch_w_h = get_params(shape_w_h, T=True)
    walnut_b_h, torch_b_h = get_params(shape_b_h)

    walnut_module = walnut.nn.blocks.Recurrent(Cin, Ch, num_layers=1)
    walnut_module.training = True
    walnut_module.sub_modules[0].w = walnut_w_in
    walnut_module.sub_modules[0].b = walnut_b_in
    walnut_module.sub_modules[1].w = walnut_w_h
    walnut_module.sub_modules[1].b = walnut_b_h
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.RNN(Cin, Ch, batch_first=True, num_layers=1)
    torch_module.weight_ih_l0 = torch_w_in
    torch_module.bias_ih_l0 = torch_b_in
    torch_module.weight_hh_l0 = torch_w_h
    torch_module.bias_hh_l0 = torch_b_h
    torch_y = torch_module(torch_x)[0]

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False)
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    acc = 1e-4  # inaccuracies?

    results.append(validate(walnut_dx, torch_x.grad, acc))
    results.append(
        validate(
            walnut_module.sub_modules[0].w.grad, torch_module.weight_ih_l0.grad.T, acc
        )
    )
    results.append(
        validate(walnut_module.sub_modules[0].b.grad, torch_module.bias_ih_l0.grad, acc)
    )
    results.append(
        validate(
            walnut_module.sub_modules[1].w.grad, torch_module.weight_hh_l0.grad.T, acc
        )
    )
    results.append(
        validate(walnut_module.sub_modules[1].b.grad, torch_module.bias_hh_l0.grad, acc)
    )

    assert all(results)


def test_recurrent_cuda() -> None:
    results = []
    shape_x = (B, X, Cin)
    shape_w_in = (Cin, Ch)
    shape_b_in = (Ch,)
    shape_w_h = (Ch, Ch)
    shape_b_h = (Ch,)

    # forward
    walnut_x, torch_x = get_vals(shape_x, device="cuda")
    walnut_w_in, torch_w_in = get_params(shape_w_in, T=True, device="cuda")
    walnut_b_in, torch_b_in = get_params(shape_b_in, device="cuda")
    walnut_w_h, torch_w_h = get_params(shape_w_h, T=True, device="cuda")
    walnut_b_h, torch_b_h = get_params(shape_b_h, device="cuda")

    walnut_module = walnut.nn.blocks.Recurrent(Cin, Ch, num_layers=1)
    walnut_module.training = True
    walnut_module.sub_modules[0].w = walnut_w_in
    walnut_module.sub_modules[0].b = walnut_b_in
    walnut_module.sub_modules[1].w = walnut_w_h
    walnut_module.sub_modules[1].b = walnut_b_h
    walnut_module.to_device("cuda")
    walnut_y = walnut_module(walnut_x)

    torch_module = torch.nn.RNN(Cin, Ch, batch_first=True, num_layers=1)
    torch_module.weight_ih_l0 = torch_w_in
    torch_module.bias_ih_l0 = torch_b_in
    torch_module.weight_hh_l0 = torch_w_h
    torch_module.bias_hh_l0 = torch_b_h
    torch_y = torch_module(torch_x)[0]

    results.append(validate(walnut_y, torch_y))

    # backward
    walnut_dy, torch_dy = get_vals(walnut_y.shape, torch_grad=False, device="cuda")
    walnut_dx = walnut_module.backward(walnut_dy.data)
    torch_y.backward(torch_dy)

    acc = 1e-4  # inaccuracies?

    results.append(validate(walnut_dx, torch_x.grad, acc))
    results.append(
        validate(
            walnut_module.sub_modules[0].w.grad, torch_module.weight_ih_l0.grad.T, acc
        )
    )
    results.append(
        validate(walnut_module.sub_modules[0].b.grad, torch_module.bias_ih_l0.grad, acc)
    )
    results.append(
        validate(
            walnut_module.sub_modules[1].w.grad, torch_module.weight_hh_l0.grad.T, acc
        )
    )
    results.append(
        validate(walnut_module.sub_modules[1].b.grad, torch_module.bias_hh_l0.grad, acc)
    )

    assert all(results)
