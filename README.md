# Numpy Neural Network

This is a framework for building neural networks, train them and analyzing the results created using NumPy only.

Models can be buit using a variety of layers, such as a linear or convolutional layer. Most of the common activation functions can be used.

```python
model = networks.FeedForward(input_shape=(4,), layers=[
    layers.Linear(nr_neurons=16, activation=activations.Tanh(), init=inits.Kaiming),
    layers.Linear(nr_neurons=16, activation=activations.Tanh(), init=inits.Kaiming),
    layers.Linear(nr_neurons=16, activation=activations.Tanh(), init=inits.Kaiming),
    layers.Linear(nr_neurons=3, activation=activations.Softmax(), init=inits.Kaiming)
])
```

The model can be trained using common algorithms, such as SGD or adam.

```python
model.compile(
    optimizer=optimizers.sgd(learning_rate=0.01, momentum=0.9, nesterov=True),
    loss_function=losses.crossentropy(),
    metric=metrics.Accuracy
)
```

The framework also provides some functions for analyszing a models' parameters and gradients to gain some insights (inspired by Andrej Karpathy :) )

![image](https://github.com/DKoflerGIT/NumpyNN/assets/74835806/a205f974-40a6-4d7b-9916-060d4ada9cae)

![image](https://github.com/DKoflerGIT/NumpyNN/assets/74835806/8119d55a-fb83-4300-8f9f-5ea1bd8e85d1)

Individual layers can also be used without a network.

```python
cnv = layers.Convolution(nr_kernels=1, kernel_size=(3, 3))
cnv.x = a
cnv.w = w
cnv.b = b
cnv.forward()
```

This project is still a work in progress, as I am planning to constantly add new features and optimizations.
The code is not perfect, as I am still learning with every new feature that is implement.
If you have any suggestions or find any bugs, please don't hesitate to contact me.
