# Compyute Neural Networks

`Compyute` is a toolbox for building, training and analyzing neural networks. This framework was developed using `NumPy`/`CuPy` only.

## Installation

All you need is to pip install the requirements (`pip install -r requirements.txt`). As of `CuPy` v13, the package does not require a GPU toolkit to be installed, so `Compyute` can also be used on CPU-only machines. If you want to make use of GPUs, make sure to install the CUDA Toolkit.

## The Toolbox

There are example-notebooks included that show how to use the toolbox and its `Tensor` object.

### Tensors
Similar to `PyTorch`, in `Compyute` a `Tensor`-object represents the central block that keeps track of data and it's gradients. However, unlike `PyTorch`, this toolbox does not support autograd to compute gradients. Instead the computation of gradients is defined within a model's layers. The `Tensor` object supports most operations also known from `PyTorch` tensors or `NumPy` arrays.

```python
# create a tensor from a list of lists, the data type is inferred automatically
a = Tensor([[4, 5, 6], [7, 8, 9]])

# alternatively, define data types
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
f = a.sum(axis=0)
```

### Data preprocessing, Encoding
The framework offers some utility functions to preprocess and encode data before using it for training (e.g. tokenizers).

```python
from compyute.preprocessing.text import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.fit(data, vocab_size=400)
```

### Building and training models
Models can be built using predefined model-templates, like the `SequentialModel`.

```python
import compyute.nn as nn

model = nn.SequentialModel([
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Batchnorm1d(16),
    nn.Linear(16, 3),
])
```

Alternatively, models can also be built entirely from scratch, using custom classes that inherit from the `Model` class. With custom models, the user defines what modules to use and how data and gradients flow through the network. Models are generally composed of one or more `Modules` (e.g. layers in a `SequentialModel`). `Compyute` provides a variety of modules such as activation, normalization, linear, convolutional and recurrent layers with more to come. 

```python
import compyute.nn as nn

class MyCustomModel(nn.Model):
    def __init__(self):
        super().__init__()

        # define your layers
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
        self.backward = backward
        
        return x

model = MyCustomModel()
```

Defined models can be trained and updated using common optimizer algorithmes, such as SGD or Adam. Models can also be saved and loaded later on.

```python
from compyute.nn.trainer import Trainer

trainer = Trainer(
    optimizer="sgd",
    loss_function="crossentropy",
    metric_function="accuracy"
)
trainer.train(X_train, y_train, epochs=10)

nn.save_model(model, "my_model.cp")
```

## License
This code base is available under the MIT License.

## Final Notes
I use this project to gain a deeper understanding of the inner workings of neural networks. Therefore, project is a work in progress and possibly will forever be, as I am planning to constantly add new features and optimizations. The code is by far not perfect, as I am still learning with every new feature that is added. If you have any suggestions or find any bugs, please don't hesitate to contact me.

Cheers,<br>
Daniel
