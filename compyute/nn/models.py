"""Neural network models module"""

import pickle
from .modules.containers import Sequential
from .dataloaders import DataLoader
from .modules.module import Module
from ..functional import concatenate, ones
from ..tensor import Tensor
from ..types import DtypeLike, ShapeLike


__all__ = ["Model", "SequentialModel", "save_model", "load_model", "model_summary"]


class Model(Module):
    """Trainable neural network model."""

    def __init__(self, core_module: Module | None = None) -> None:
        """Trainable neural network model.

        Parameters
        ----------
        core_module : Module | None, optional
            Core module of the model. For multiple modules use a container as core module.
        """
        super().__init__()
        self.core_module = core_module

    @property
    def child_modules(self) -> list[Module]:
        """Model child modules."""
        if self.core_module is not None:
            return [self.core_module]
        return [i[1] for i in self.__dict__.items() if isinstance(i[1], Module)]

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the module.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        ----------
        Tensor
            Computed module output.
        """
        if self.core_module is None:
            raise ValueError("Forward function is not defined.")

        y = self.core_module.forward(x)

        if self.training:
            self.backward = self.core_module.backward

        return y

    def predict(self, X: Tensor, batch_size: int = 1) -> Tensor:
        """Returns the models predictions for a given input.

        Parameters
        ----------
        X : Tensor
            Input tensor.
        batch_size : int, optional
            Number of inputs processed in parallel, by default 1.

        Returns
        -------
        Tensor
            Predictions.
        """

        dataloader = DataLoader(X, None, batch_size)
        outputs = []

        for batch in dataloader(shuffle_data=False):
            X_batch = batch[0]
            X_batch.to_device(self.device)
            outputs.append(self.forward(X_batch))

        if not self.retain_values:
            self.reset()

        return concatenate(outputs, axis=0)


class SequentialModel(Model):
    """Sequential model. Layers are processed sequentially."""

    def __init__(self, layers: list[Module]) -> None:
        """Sequential model. Layers are processed sequentially.

        Parameters
        ----------
        layers : list[Module]
            List of layers for the model.
            These layers are processed sequentially starting at index 0.
        """
        super().__init__(Sequential(layers))


def save_model(model: Model, filepath: str) -> None:
    """Saves a model as a binary file.

    Parameters
    ----------
    model : Model
        Model to be saved.
    filepath : str
        Path to the file.
    """

    model.to_device("cpu")
    model.reset()

    file = open(filepath, "wb")
    pickle.dump(model, file)
    file.close()


def load_model(filepath: str) -> Model:
    """Load a model from a previously saved binary file.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    Model
        Loaded model.
    """
    file = open(filepath, "rb")
    obj = pickle.load(file)
    file.close()
    return obj


def model_summary(
    model: Model, input_shape: ShapeLike, input_dtype: DtypeLike = "float32"
) -> None:
    """Prints information about a model.

    Parameters
    ----------
    model : Model
        Neural network model.
    input_shape : ShapeLike
        Shape of the model input ignoring the batch dimension.
    input_dtype : DtypeLike
        Data type of the expected input data.
    """
    n = 63

    summary = [f"{model.__class__.__name__}\n{'-' * n}"]
    summary.append(f"\n{'Layer':25s} {'Output Shape':20s} {'# Parameters':>15s}\n")
    summary.append("=" * n)
    summary.append("\n")

    x = ones((1,) + input_shape, dtype=input_dtype)
    x.to_device(model.device)
    retain_values = model.retain_values
    model.retain_values = True
    _ = model.forward(x)

    def build_string(module, summary, depth):
        if not isinstance(module, Model):
            name = " " * depth + module.__class__.__name__
            output_shape = str((-1,) + module.y.shape[1:])
            n_params = sum(p.size for p in module.parameters)
            summary.append(f"{name:25s} {output_shape:20s} {n_params:15d}\n")

        for module in module.child_modules:
            build_string(module, summary, depth + 1)

    build_string(model, summary, -1)
    summary.append("=" * n)
    n_parameters = sum(p.data.size for p in model.parameters)

    model.reset()
    model.retain_values = retain_values
    string = "".join(summary)
    print(f"{string}\n\nTotal parameters: {n_parameters}")
