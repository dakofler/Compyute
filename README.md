# Walnut

`Walnut` is a toolbox for building, training and analyzing neural networks. This framework was developed using `NumPy`/`CuPy` only. There are example-notebooks that explain how to use the framework and its `Tensor` objects.

Unfortunately, installing `CuPy` is required for now, even if no GPU is available.

## Installation

All you need to use the library is to pip install the requirements (`pip install -r requirements.txt`). As of `CuPy` v13, it does not require a GPU toolkit to be installed, so `Walnut` can now be used on CPU-only machines also. If you want to make use of GPUs, make sure to install CUDA/ROCm.

## The Toolbox

There are examples included that show how to use the toolbox.

### Tensors
Similar to `PyTorch`, in `Walnut` a `Tensor`-object represents the central block that keeps track of data and it's gradients. However, unlike `PyTorch`, this toolbox does not support autograd to compute gradients. Instead the computation of gradients happens within a model's layers (modules). The `Tensor` object supports most operations also known from `PyTorch` tensors or `NumPy` arrays. `Walnut` also supports CUDA.

### Data preprocessing, Encoding
The framework offers some utility functions to preprocess and encode data before using it for training (e.g. tokenizers).

### Building and training models
Models can be built using predefined model-templates (e.g. `Sequential`), or they can also be built entirely from scratch, using custom classes that inherit from the `Model` class. With custom models, the user defines what modules to use and how data and gradients flow through the network. Models are generally composed of one or more `Modules` (e.g. layers in a `Sequential` model.). `Walnut` provides a variety of modules such as activation, normalization, linear, convolutional and recurrent layers with more to come. Defined models can be trained and updated using common optimizer algorithmes, such as SGD or Adam. Models can also be saved and loaded later on.

### Analysis
`Walnut` also provides some functions for analyzing a models' paramenters, outputs and gradients to gain insights and help understand the effects of hyperparameters (inspired by Andrej Karpathy's YouTube videos - cannot recommend them enough).

### Experimenting
All modules (layers) can also be used and experimented with individually without a model.

## License
This code base is available under the MIT License.

## Final Notes
I use this project to gain a deeper understanding of the inner workings of neural networks. Therefore project is a work in progress and possibly will forever be, as I am planning to constantly add new features and optimizations.
The code is by far not perfect, as I am still learning with every new feature that is added. If you have any suggestions or find any bugs, please don't hesitate to contact me.

Cheers,<br>
Daniel


