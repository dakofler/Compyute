# Numpy Neural Network

This is a framework for building neural networks, train them and analyzing the results created using NumPy only.

Models can be buit using a variety of layers, such as a linear or convolutional layer. Most of the common activation functions can be used.

```python
model = networks.Sequential(input_shape=(4,), layers=[
    layers.Linear(out_channels=16, act_fn=activations.Tanh(), init_fn=inits.Kaiming),
    layers.Linear(out_channels=16, act_fn=activations.Tanh(), init_fn=inits.Kaiming),
    layers.Linear(out_channels=16, act_fn=activations.Tanh(), init_fn=inits.Kaiming),
    layers.Linear(out_channels=3, act_fn=activations.Softmax(), init_fn=inits.Kaiming)
])
```

```python
model = networks.Sequential(input_shape=(1, 28, 28), layers=[
    layers.Convolution(out_channels=8, kernel_shape=(3, 3), act_fn=activations.Relu()),
    layers.MaxPooling(p_window=(2, 2)),
    layers.Convolution(out_channels=16, kernel_shape=(3, 3), act_fn=activations.Relu()),
    layers.MaxPooling(p_window=(2, 2)),
    layers.Flatten(),
    layers.Linear(out_channels=64, act_fn=activations.Relu()),
    layers.Linear(out_channels=10, act_fn=activations.Softmax())
])
```

The model can be trained using common algorithms, such as SGD or adam.

```python
model.compile(
    optimizer=optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True),
    loss_fn=losses.Crossentropy(),
    metric=metrics.accuracy
)
```

The framework also provides some functions for analyszing a models' parameters and gradients to gain some insights (inspired by Andrej Karpathy :) )

![image](https://github.com/DKoflerGIT/NumpyNN/assets/74835806/a205f974-40a6-4d7b-9916-060d4ada9cae)

![image](https://github.com/DKoflerGIT/NumpyNN/assets/74835806/8119d55a-fb83-4300-8f9f-5ea1bd8e85d1)

This project is still a work in progress, as I am planning to constantly add new features and optimizations.
The code is not perfect, as I am still learning with every new feature that is implement.
If you have any suggestions or find any bugs, please don't hesitate to contact me.
