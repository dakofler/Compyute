# Walnut

`Walnut` is a toolbox for building, training and analyzing neural networks. This framework was developed using `NumPy` only. There are example-notebook that explain how to use the framework and it's `Tensor` objects.

## Tensors
Similar to `PyTorch` a `Tensor`-object was introduced as the central block that keeps track of data, gradients and more. Unlike `PyTorch`, this framework does not support autograd. Gradients used for training neural networks are computed within the network's layers. The tensors ogject supporst most operations also known from `PyTorch`'s tensors or `NumPy` arrays.

## Neural Networks

### Data preprocessing, Encoding
The framework offers some utility functions to preprocess and encode data before using it for training.

### Designing a model
Models can be built using predefined model-templates (e.g. a `Sequential`), but can also be built from scratch. They are generally composed of one or more modules (e.g. layers in a `Sequential` model.). The framework provides a variety of layers such as activation, normalization, linear and convolutional layers with more to come.

### Training a model
The model can be trained using common optimizer algorithmes, such as SGD or Adam.

### Analysis
The framework also provides some functions for analyzing a models' parameters and gradients to gain insights (inspired by Andrej Karpathy :) )

### Experimenting
All modules (layers) can also be used individually without a model and their parameters can be inspected.

This project is still a work in progress, as I am planning to constantly add new features and optimizations.
The code is not perfect, as I am still learning with every new feature that is added.
If you have any suggestions or find any bugs, please don't hesitate to contact me.
