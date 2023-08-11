# Walnut
This is a framework for working with tensors and building, training and analyzing neural networks. This framework was developed using NumPy only. I have provided some example-notebook that explain how to use the framework and it's tensors.

## Tensors
Similar to PyTorch I introduced a `Tensor`-object as the central building block that keeps track of its data, gradients and more. However, this framework does not support autograd like pytorch does. Gradients used for neural networks are computed within a network's layer. Tensors support most operations also known from Pytorch tensors or Numpy arrays.

## Neural Networks

### Data preprocessing
The framework offers some utility functions, such as `split_train_val_test()` to preprocess data before using it for training.

### Designing a model
Models can be built using predefined model-templates (e.g. a sequential model), but can also be built from scratch. Models are generally composed of one or more modules (e.g. layers in a sequential model.). The framework provides a variety of layers such as activation, normalization, linear and convolutional layers.

### Training a model
The model can be trained using common optimizer algorithmes, such as SGD or Adam.

### Analysis
The framework also provides some functions for analyzing a models' parameters and gradients to gain insights (inspired by Andrej Karpathy :) )

### Experimenting
All modules (layers) can also be used individually without a model and their parameters can be inspected.

This project is still a work in progress, as I am planning to constantly add new features and optimizations.
The code is not perfect, as I am still learning with every new feature that is added.
If you have any suggestions or find any bugs, please don't hesitate to contact me.