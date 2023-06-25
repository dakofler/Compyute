# Walnut

This is a framework for working with tensors and building, training and analyzing neural networks created using NumPy only. Similar to PyTorch I introduced a `Tensor`-object as the central building block that keeps track of its data, gradients and more. However, this framework does not support autograd. Gradients are computed within a network's layer.

```Python
import walnut

a = walnut.Tensor([1, 2, 3])
b = walnut.randn((3,))
c = a + b # addition of tensors
d = a @ b # matrix multiplication of tensors
e = walnut.zeros((5, 5))
```

### Data preprocessing

The framework offers some utility functions, such as `split_train_val_test()` to preprocess data before using it for training.

### Designing a model

Models can be built using a variety of layers, including trainable layers such as linear or convolutional layers. Most of the common activation functions as well as normalization can be applied.

```python
import walnut.nn as nn

model = nn.Sequential(mdl_layers=[
    nn.layers.Linear(input_shape=(4,), out_channels=16, act_fn=nn.activations.Tanh(), init_fn=nn.inits.Kaiming),
    nn.layers.Linear(out_channels=16, act_fn=nn.activations.Tanh(), init_fn=nn.inits.Kaiming),
    nn.layers.Linear(out_channels=16, act_fn=nn.activations.Tanh(), init_fn=nn.inits.Kaiming),
    nn.layers.Linear(out_channels=3, act_fn=nn.activations.Softmax(), init_fn=nn.inits.Kaiming)
])
```

```python
model = nn.Sequential(mdl_layers=[
    nn.layers.Convolution(input_shape=(1, 28, 28), out_channels=8, kernel_shape=(3, 3), act_fn=nn.activations.Relu(), norm_fn=nn.norms.Layernorm()),
    nn.layers.MaxPooling(p_window=(2, 2)),
    nn.layers.Convolution(out_channels=16, kernel_shape=(3, 3), act_fn=nn.activations.Relu(), norm_fn=nn.norms.Layernorm()),
    nn.layers.MaxPooling(p_window=(2, 2)),
    nn.layers.Flatten(),
    nn.layers.Linear(out_channels=64, act_fn=nn.activations.Relu(), norm_fn=nn.norms.Layernorm()),
    nn.layers.Linear(out_channels=10, act_fn=nn.activations.Softmax())
])
```

### Training a model

The model can be trained using common algorithms, such as SGD or Adam.

```python
model.compile(
    optimizer=nn.optimizers.SGD(l_r=1e-2, momentum=0.9, nesterov=True),
    loss_fn=nn.losses.Crossentropy(),
    metric=nn.metrics.accuracy
)
```

```python
model.compile(
    optimizer=nn.optimizers.Adam(l_r=1e-3),
    loss_fn=nn.losses.Crossentropy(),
    metric=nn.metrics.accuracy
)
```

```python
hist = model.train(x_train, y_train, epochs=10, batch_size=512, val_data=(x_val, y_val))
model.plot_training_loss(hist)
```

### Analysis

The framework also provides some functions for analyzing a models' parameters and gradients to gain insights (inspired by Andrej Karpathy :) )

![image](https://github.com/DKoflerGIT/NumpyNN/assets/74835806/a205f974-40a6-4d7b-9916-060d4ada9cae)

![image](https://github.com/DKoflerGIT/NumpyNN/assets/74835806/8119d55a-fb83-4300-8f9f-5ea1bd8e85d1)


### Experimenting

All layers can also be used individually without a model and their parameters can be inspected.
```python
conv = nn.layers.Convolution(out_channels=2)

conv.x = X
conv.w = W
conv.b = B

conv.forward()
conv.backward()

conv.x.grad
conv.w.grad
conv.b.grad
```

This project is still a work in progress, as I am planning to constantly add new features and optimizations.
The code is not perfect, as I am still learning with every new feature that is implement.
If you have any suggestions or find any bugs, please don't hesitate to contact me.