# Compyute Neural Networks

[![CI/CD](https://github.com/dakofler/Compyute/actions/workflows/tests.yml/badge.svg)](https://github.com/dakofler/Compyute/actions/workflows/tests.yml)

`Compyute` is a toolbox for building and training and analyzing neural networks only using `NumPy`/`CuPy` under the hood to perform computations.

## Installation

All you need is to pip install .
```bash
git clone https://github.com/dakofler/Compyute
pip install .
```
As of `CuPy` v13, the package does not require a GPU toolkit to be installed, so `Compyute` can be used on CPU-only machines. If you want to make use of GPUs, make sure to install the CUDA Toolkit following the installation guide of `CuPy` (https://docs.cupy.dev/en/stable/install.html).

## Usage

There are example-notebooks included that show how to use the toolbox and its `Tensor` object.

Similar to `PyTorch`, in `Compyute` a `Tensor`-object represents the central block that keeps track of data and gradients. However, unlike `PyTorch`, `Compyute` does not support autograd to compute gradients (yet?). Instead the computation of gradients is defined within a model's layers and functions.

### Tensors

```python
import compyute as cp

# create a tensor from a list of lists, the data type is inferred automatically
x = cp.tensor([[4, 5, 6], [7, 8, 9]])

# alternatively, define data types
x = cp.tensor([1, 2, 3], dtype=cp.int32)
x = cp.tensor([1, 2, 3], dtype="int64")

# change datatypes
y = x.as_type(cp.float32)
y = x.float()
y = x.int()

# define the device the tensor is stored on
c = cp.tensor([1, 2, 3], device=cp.cuda)
c = cp.tensor([1, 2, 3], device="cuda")

# change devices
c.to_device(cp.cpu)
c.to_device("cpu")
c = c.cpu()
```

The `Tensor` object supports most operations also known from `PyTorch` tensors or `NumPy` arrays. Here are some examples:

```python
# basic arithmetic
z = x + y
z = x - y
z = x * y
z = x / y
x += 10

# operations from linear algebra
z = x @ y
z = cp.inner(x, y)
z = cp.outer(x, y)

# aggregate elements of a tensor
z = cp.sum(x, axis=0)
z = cp.mean(x, axis=0)

# functions for initializing tensors
z = cp.ones(shape=(10, 10))
z = cp.zeros(shape=(10, 10))
z = cp.full(shape=(10, 10), value=99)
```

### Data preprocessing, Encoding
The framework offers some utility functions to preprocess ...

```python
from compyute.preprocessing import split_train_val_test

train, val, test = split_train_val_test(x, ratio_val=0.25, ratio_test=0.25)
```

... and encode data before using it for training.

```python
from compyute.preprocessing.text import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.fit(data, vocab_size=400)
```

### Building and training models
Models can be built using modules and containers, like `Sequential`. This also allows you to define your own blocks and reuse them in models.

```python
import compyute.nn as nn

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)
```

Alternatively, models can also be built entirely from scratch by using custom classes that inherit from the `Container` class. With custom models, the user defines what modules to use and how data and gradients flow through the network. Models are generally composed of one or more `Modules` (e.g. layers in a `Sequential`). `Compyute` provides a variety of modules such as activation, normalization, linear, convolutional and recurrent layers with more to come. 

```python
import compyute.nn as nn

# create a block or model by inheriting from the 'Sequential' container
class MyConvBlock(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Convolution2D(32, 64, kernel_size=3),
            nn.ReLU(),
            Batchnorm1d(16)
        )

# create a model from scratch by inheriting from the 'Container' base class
class MyModel(nn.Container):
    def __init__(self):
        super().__init__()

        # define your modules
        self.lin1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.bn = nn.Batchnorm1d(16)
        self.lin2 = nn.Linear(16, 3)

    def forward(self, x):
        # define the forward pass
        x = self.lin1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.lin2(x)

        # define the backward pass
        def backward(dy):
            dy = self.lin2.backward(dy)
            dy = self.bn.backward(dy)
            dy = self.relu.backward(dy)
            dy = self.lin1.backward(dy)
            return dy

        self._backward = backward
        
        return x

my_model = MyModel()
```

All modules can be trained and updated using common optimizer algorithms, such as SGD or Adam. Callbacks offer extended functionality such as tracking the loss-values during training.

```python
from compyute.nn.trainer import Trainer
from compyute.nn.callbacks import EarlyStopping, History, Progressbar

# define trainer
trainer = Trainer(
    model=my_model,
    optimizer="sgd",
    loss_function="crossentropy",
    metric_function="accuracy",
    callbacks=[EarlyStopping(), History(), Progressbar()]
)

# train model
trainer.train(X_train, y_train, epochs=10)
```

Alternatively, you can write your own training loop.

```python
epochs = 100
batch_size = 32

train_dl = nn.DataLoader(X_train, y_train, batch_size)
val_dl = nn.DataLoader(X_val, y_val, batch_size)
loss_func = nn.CrossEntropy()
optim = nn.optimizers.SGD(model.parameters)

for epoch in range(epochs):
    # training
    with model.training():
        for x, y in train_dl():
            # forward pass
            y_pred = model(x)
            _ = loss_func(y_pred, y)

            # backward pass
            optim.reset_grads()  # reset all gradients
            model.backward(loss_func.backward())  # compute new gradients
            optim.step()  # update parameters
    
    # validiation
    val_loss = 0
    for x, y in val_dl():
        y_pred = model(x)
        val_loss += loss_func(y_pred, y).item()
    val_loss /= len(val_dl)
    print(f"epoch {epoch}: {val_loss=:.4f}")
```

Models can also be saved and loaded later on.

```python
nn.save_module(model, "my_model.cp")
```

## Author
Daniel Kofler - AI Research Associate ([dkofler@outlook.com](mailto:dkofler@outlook.com))

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Final Notes
I use this project to gain a deeper understanding of the inner workings of neural networks. Therefore, project is a work in progress and possibly will forever be, as I am planning to constantly add new features and optimizations. The code is by far not perfect, as I am still learning with every new feature that is added. If you have any suggestions or find any bugs, please don't hesitate to contact me.

Cheers,<br>
Daniel
