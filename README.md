# Numpy Neural Network

This is a framework for building, training and analyzing neural networks created using NumPy only. Similar to PyTorch I introduced a `Tensor`-object as the central building block that keeps track of its data, gradients and more. However, this framework does not support autograd. Gradients are computed within a network's layer.

```Python
a = Tensor([1, 2, 3])
b = nn.randn((3,))

c = a + b

d = nn.tensor.zeros((5, 5))
```

### Data preprocessing

The framework offers some utility functions, such as `split_train_val_test()` to preprocess data before using it for training.

### Designing a model

Models can be built using a variety of layers, including trainable layers such as linear or convolutional layers. Most of the common activation functions as well as normalization can be applied.

```python
model = networks.Sequential(input_shape=(4,), mdl_layers=[
    layers.Linear(out_channels=16, act_fn=activations.Tanh(), init_fn=inits.Kaiming),
    layers.Linear(out_channels=16, act_fn=activations.Tanh(), init_fn=inits.Kaiming),
    layers.Linear(out_channels=16, act_fn=activations.Tanh(), init_fn=inits.Kaiming),
    layers.Linear(out_channels=3, act_fn=activations.Softmax(), init_fn=inits.Kaiming)
])
```

```python
model = networks.Sequential(input_shape=(1, 28, 28), mdl_layers=[
    layers.Convolution(out_channels=8, kernel_shape=(3, 3), act_fn=activations.Relu(), norm_fn=norms.Layernorm()),
    layers.MaxPooling(p_window=(2, 2)),
    layers.Convolution(out_channels=16, kernel_shape=(3, 3), act_fn=activations.Relu(), norm_fn=norms.Layernorm()),
    layers.MaxPooling(p_window=(2, 2)),
    layers.Flatten(),
    layers.Linear(out_channels=64, act_fn=activations.Relu(), norm_fn=norms.Layernorm()),
    layers.Linear(out_channels=10, act_fn=activations.Softmax())
])
```

### Training a model

The model can be trained using common algorithms, such as SGD or Adam.

```python
model.compile(
    optimizer=optimizers.SGD(l_r=1e-2, momentum=0.9, nesterov=True),
    loss_fn=losses.Crossentropy(),
    metric=metrics.accuracy
)
```

```python
model.compile(
    optimizer=optimizers.Adam(l_r=1e-3),
    loss_fn=losses.Crossentropy(),
    metric=metrics.accuracy
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
conv = layers.Convolution(out_channels=2)

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