"""Neural network models module"""

import pickle
from .modules.containers import SequentialContainer
from .dataloaders import DataLoader
from .modules.module import Module
from ..functional import concatenate, ones
from ..tensor import Tensor
from ..types import DtypeLike, ShapeLike


__all__ = ["Model", "SequentialModel", "save_model", "load_model"]


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

        dataloader = DataLoader(X, batch_size=batch_size)
        outputs = []

        for x in dataloader(shuffle=False):
            x.to_device(self.device)
            outputs.append(self.forward(x))

        if not self.retain_values:
            self.reset()

        return concatenate(outputs, axis=0)

    def summary(
        self, input_shape: ShapeLike, input_dtype: DtypeLike = "float32"
    ) -> None:
        """Prints information about the model.

        Parameters
        ----------
        input_shape : ShapeLike
            Shape of the model input ignoring the batch dimension.
        input_dtype : DtypeLike
            Data type of the expected input data.
        """
        n = 63

        summary = [f"{self.__class__.__name__}\n{'-' * n}"]
        summary += [f"\n{'Layer':25s} {'Output Shape':20s} {'# Parameters':>15s}\n"]
        summary += ["=" * n, "\n"]

        x = ones((1,) + input_shape, dtype=input_dtype, device=self.device)
        retain_values = self.retain_values
        self.retain_values = True
        _ = self.forward(x)

        def build_summary(module, summary, depth):
            if not isinstance(module, Model):
                name = " " * depth + module.__class__.__name__
                output_shape = str((-1,) + module.y.shape[1:])
                n_params = sum(p.size for p in module.parameters)
                summary += [f"{name:25s} {output_shape:20s} {n_params:15d}\n"]

            if module.child_modules is not None:
                for module in module.child_modules:
                    build_summary(module, summary, depth + 1)

        build_summary(self, summary, -1)
        summary += ["=" * n]
        n_parameters = sum(p.size for p in self.parameters)

        self.reset()
        self.retain_values = retain_values
        summary = "".join(summary)
        print(f"{summary}\n\nTotal parameters: {n_parameters}")


class SequentialModel(Model):
    """Sequential model. Layers are processed sequentially."""

    def __init__(self, layers: list[Module] | None = None) -> None:
        """Sequential model. Layers are processed sequentially.

        Parameters
        ----------
        layers : list[Module] | None, optional
            List of layers for the model.
            These layers are processed sequentially starting at index 0.
        """
        super().__init__(SequentialContainer(layers))

    def add(self, layer: Module) -> None:
        """Adds a layer to the model.

        Parameters
        ----------
        layer : Module
            Layer to append.
        """
        if self.core_module.child_modules is None:
            self.core_module.child_modules = [layer]
        else:
            self.core_module.child_modules.append(layer)


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
