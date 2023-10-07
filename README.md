# Walnut

`Walnut` is a toolbox for building, training and analyzing neural networks. This framework was developed using `NumPy`/`CuPy` only. There are example-notebook that explain how to use the framework and it's `Tensor` objects.

## Tensors
Similar to `PyTorch`, in `Walnut` a `Tensor`-object represents the central block that keeps track of data and it's gradients. However, unlike `PyTorch`, this toolbox does not support autograd to compute gradients. Instead the computation of gradients happens within a model's layers (modules). The `Tensor` object supports most operations also known from `PyTorch` tensors or `NumPy` arrays. `Walnut` also supports CUDA.

## Neural Networks

### Data preprocessing, Encoding
The framework offers some utility functions to preprocess and encode data before using it for training (e.g. tokenizers).

### Building and training models
Models can be built using predefined model-templates (e.g. `Sequential`), or they can also be built entirely from scratch, using custom classes that inherit from the `Model` class. With custom models, the user defines what modules to use and how data and gradients flow through the network. Models are generally composed of one or more `Modules` (e.g. layers in a `Sequential` model.). `Walnut` provides a variety of modules such as activation, normalization, linear, convolutional and recurrent layers with more to come. Defined models can be trained and updated using common optimizer algorithmes, such as SGD or Adam. Models can also be saved and loaded later on.

### Analysis
`Walnut` also provides some functions for analyzing a models' paramenters, outputs and gradients to gain insights and help understand the effects of hyperparameters (inspired by Andrej Karpathy :) )

### Experimenting
All modules (layers) can also be used and experimented with individually without a model.


This project is still a work in progress, as I am planning to constantly add new features and optimizations.
The code is not perfect, as I am still learning with every new feature that is added.
If you have any suggestions or find any bugs, please don't hesitate to contact me.
