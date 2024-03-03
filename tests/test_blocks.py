"""block tests"""

import torch
import compyute
from tests.test_utils import get_vals, get_params, validate


B, Cin, Ch, Cout, X = (10, 10, 10, 10, 10)


# Recurrent
def test_recurrent_cpu() -> None:
    results = []
    shape_x = (B, X, Cin)
    shape_w_in = (Cin, Ch)
    shape_b_in = (Ch,)
    shape_w_h = (Ch, Ch)
    shape_b_h = (Ch,)

    # forward
    compyute_x, torch_x = get_vals(shape_x)
    compyute_w_in, torch_w_in = get_params(shape_w_in, T=True)
    compyute_b_in, torch_b_in = get_params(shape_b_in)
    compyute_w_h, torch_w_h = get_params(shape_w_h, T=True)
    compyute_b_h, torch_b_h = get_params(shape_b_h)

    compyute_module = compyute.nn.blocks.Recurrent(Cin, Ch, num_layers=1)
    compyute_module.training = True
    compyute_module.child_modules[0].w = compyute_w_in
    compyute_module.child_modules[0].b = compyute_b_in
    compyute_module.child_modules[1].w = compyute_w_h
    compyute_module.child_modules[1].b = compyute_b_h
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.RNN(Cin, Ch, batch_first=True, num_layers=1)
    torch_module.weight_ih_l0 = torch_w_in
    torch_module.bias_ih_l0 = torch_b_in
    torch_module.weight_hh_l0 = torch_w_h
    torch_module.bias_hh_l0 = torch_b_h
    torch_y = torch_module(torch_x)[0]

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)

    acc = 1e-4  # inaccuracies?

    results.append(validate(compyute_dx, torch_x.grad, acc))
    results.append(
        validate(
            compyute_module.child_modules[0].w.grad,
            torch_module.weight_ih_l0.grad.T,
            acc,
        )
    )
    results.append(
        validate(
            compyute_module.child_modules[0].b.grad, torch_module.bias_ih_l0.grad, acc
        )
    )
    results.append(
        validate(
            compyute_module.child_modules[1].w.grad,
            torch_module.weight_hh_l0.grad.T,
            acc,
        )
    )
    results.append(
        validate(
            compyute_module.child_modules[1].b.grad, torch_module.bias_hh_l0.grad, acc
        )
    )

    assert all(results)


def test_recurrent_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []
    shape_x = (B, X, Cin)
    shape_w_in = (Cin, Ch)
    shape_b_in = (Ch,)
    shape_w_h = (Ch, Ch)
    shape_b_h = (Ch,)

    # forward
    compyute_x, torch_x = get_vals(shape_x, device="cuda")
    compyute_w_in, torch_w_in = get_params(shape_w_in, T=True, device="cuda")
    compyute_b_in, torch_b_in = get_params(shape_b_in, device="cuda")
    compyute_w_h, torch_w_h = get_params(shape_w_h, T=True, device="cuda")
    compyute_b_h, torch_b_h = get_params(shape_b_h, device="cuda")

    compyute_module = compyute.nn.blocks.Recurrent(Cin, Ch, num_layers=1)
    compyute_module.training = True
    compyute_module.child_modules[0].w = compyute_w_in
    compyute_module.child_modules[0].b = compyute_b_in
    compyute_module.child_modules[1].w = compyute_w_h
    compyute_module.child_modules[1].b = compyute_b_h
    compyute_module.to_device("cuda")
    compyute_y = compyute_module(compyute_x)

    torch_module = torch.nn.RNN(Cin, Ch, batch_first=True, num_layers=1)
    torch_module.weight_ih_l0 = torch_w_in
    torch_module.bias_ih_l0 = torch_b_in
    torch_module.weight_hh_l0 = torch_w_h
    torch_module.bias_hh_l0 = torch_b_h
    torch_y = torch_module(torch_x)[0]

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False, device="cuda")
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)

    acc = 1e-3  # inaccuracies?

    results.append(validate(compyute_dx, torch_x.grad, acc))
    results.append(
        validate(
            compyute_module.child_modules[0].w.grad,
            torch_module.weight_ih_l0.grad.T,
            acc,
        )
    )
    results.append(
        validate(
            compyute_module.child_modules[0].b.grad, torch_module.bias_ih_l0.grad, acc
        )
    )
    results.append(
        validate(
            compyute_module.child_modules[1].w.grad,
            torch_module.weight_hh_l0.grad.T,
            acc,
        )
    )
    results.append(
        validate(
            compyute_module.child_modules[1].b.grad, torch_module.bias_hh_l0.grad, acc
        )
    )

    assert all(results)


# Residual
def test_residual_cpu() -> None:
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cout, Cout)

    # forward
    compyute_x, torch_x = get_vals(x_shape)
    compyute_w1, torch_w1 = get_params(w1_shape, T=True)
    compyute_w2, torch_w2 = get_params(w2_shape, T=True)

    compyute_module = compyute.nn.blocks.Residual(
        compyute.nn.containers.Sequential(
            [
                compyute.nn.layers.Linear(
                    Cin, Cout, weights=compyute_w1, use_bias=False
                ),
                compyute.nn.layers.ReLU(),
                compyute.nn.layers.Linear(
                    Cout, Cout, weights=compyute_w2, use_bias=False
                ),
            ]
        )
    )
    compyute_module.training = True
    compyute_y = compyute_module(compyute_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cout, Cout, bias=False)
    lin2.weight = torch_w2
    torch_y = torch_x + lin2(torch.nn.functional.relu(lin1(torch_x)))

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)


def test_residual_cuda() -> None:
    if not compyute.engine.gpu_available():
        pass
    results = []
    x_shape = (B, Cin)
    w1_shape = (Cin, Cout)
    w2_shape = (Cout, Cout)

    # forward
    compyute_x, torch_x = get_vals(x_shape, device="cuda")
    compyute_w1, torch_w1 = get_params(w1_shape, T=True, device="cuda")
    compyute_w2, torch_w2 = get_params(w2_shape, T=True, device="cuda")

    compyute_module = compyute.nn.blocks.Residual(
        compyute.nn.containers.Sequential(
            [
                compyute.nn.layers.Linear(
                    Cin, Cout, weights=compyute_w1, use_bias=False
                ),
                compyute.nn.layers.ReLU(),
                compyute.nn.layers.Linear(
                    Cout, Cout, weights=compyute_w2, use_bias=False
                ),
            ]
        )
    )
    compyute_module.training = True
    compyute_module.to_device("cuda")
    compyute_y = compyute_module(compyute_x)

    lin1 = torch.nn.Linear(Cin, Cout, bias=False)
    lin1.weight = torch_w1
    lin2 = torch.nn.Linear(Cout, Cout, bias=False)
    lin2.weight = torch_w2
    torch_y = torch_x + lin2(torch.nn.functional.relu(lin1(torch_x)))

    results.append(validate(compyute_y, torch_y))

    # backward
    compyute_dy, torch_dy = get_vals(compyute_y.shape, torch_grad=False, device="cuda")
    compyute_dx = compyute_module.backward(compyute_dy.data)
    torch_y.backward(torch_dy)
    results.append(validate(compyute_dx, torch_x.grad))

    assert all(results)
