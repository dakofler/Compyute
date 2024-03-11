# Compyute Neural Networks

`Compyute` is a toolbox for building, training and analyzing neural networks. This framework was developed using `NumPy`/`CuPy` only. There are example-notebooks that explain how to use the framework and its `Tensor` object.

## Installation

All you need to use the library is to pip install the requirements (`pip install -r requirements.txt`). As of `CuPy` v13, it does not require a GPU toolkit to be installed, so `Compyute` can now be used on CPU-only machines also. If you want to make use of GPUs, make sure to install CUDA.

## The Toolbox

There are examples included that show how to use the toolbox. Make sure to tweak the parameters (e.g. batch_size) to fit your available resources .

### Tensors
Similar to `PyTorch`, in `Compyute` a `Tensor`-object represents the central block that keeps track of data and it's gradients. However, unlike `PyTorch`, this toolbox does not support autograd to compute gradients. Instead the computation of gradients happens within a model's (modules). The `Tensor` object supports most operations also known from `PyTorch` tensors or `NumPy` arrays.

```python
# create a tensor from a list of lists, the data type is inferred automatically
a = Tensor([[4, 5, 6], [7, 8, 9]])

# define data types
b = Tensor([1, 2, 3], dtype="int64")

# change datatypes
b = b.float()

# define the device the tensor is stored on
c = Tensor([1, 2, 3], device="cuda")

# change devices
c.cpu()

# addition of tensors
d = a + b

# matrix multiplication of tensors
e = a @ b

# sum all elements of a tensor
f = a.sum()
```

### Data preprocessing, Encoding
The framework offers some utility functions to preprocess and encode data before using it for training (e.g. tokenizers).

### Building and training models
Models can be built using predefined model-templates (e.g. `SequentialModel`), or they can also be built entirely from scratch, using custom classes that inherit from the `Model` class. With custom models, the user defines what modules to use and how data and gradients flow through the network. Models are generally composed of one or more `Modules` (e.g. layers in a `SequentialModel`). `Compyute` provides a variety of modules such as activation, normalization, linear, convolutional and recurrent layers with more to come. Defined models can be trained and updated using common optimizer algorithmes, such as SGD or Adam. Models can also be saved and loaded later on.

```python
import compyute.nn as nn
from compyute.nn.layers import *

model = nn.SequentialModel([
    Linear(4, 16),
    ReLU(),
    Batchnorm1d(16),
    Linear(16, 3),
])

model.compile(
    optimizer=nn.optimizers.SGD(),
    loss_fn=nn.losses.Crossentropy(),
    metric_fn=nn.metrics.accuracy
)

model.train(X_train, y_train, epochs=10)

nn.save_model(model, "my_model.cp")
```

### Analysis
`Compyute` also provides some functions for analyzing a models' paramenters, outputs and gradients to gain insights and help understand the effects of hyperparameters (inspired by Andrej Karpathy's YouTube videos - cannot recommend them enough).

### Experimenting
All modules (layers) can also be used and experimented with individually without a model.

## License
This code base is available under the MIT License.

## Final Notes
I use this project to gain a deeper understanding of the inner workings of neural networks. Therefore project is a work in progress and possibly will forever be, as I am planning to constantly add new features and optimizations. The code is by far not perfect, as I am still learning with every new feature that is added. If you have any suggestions or find any bugs, please don't hesitate to contact me.

Cheers,<br>
Daniel
