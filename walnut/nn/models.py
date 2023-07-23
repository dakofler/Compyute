"""Neural network models module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ShapeLike
from walnut.logger import log_training_progress
from walnut.nn.losses import Loss
from walnut.nn.metrics import Metric
from walnut.nn.optimizers import Optimizer
from walnut.nn.modules import activations
from walnut.nn.modules import normalizations
from walnut.nn.modules.parameter import Module, ParamModule


__all__ = ["Sequential"]


class ModelCompilationError(Exception):
    """Error with the compiling of the model."""


@dataclass(repr=False)
class Model(ABC):
    """Neural network model base class."""

    loss_fn: Loss | None = None
    metric: Metric | None = None
    compiled: bool = False
    input_shape: ShapeLike = ()
    output_shape: ShapeLike = ()

    @abstractmethod
    def __call__(self, x: Tensor) -> None:
        """Calls the model."""

    @abstractmethod
    def __repr__(self) -> str:
        """Returns a string representation of the model."""

    def compile(self, loss_fn: Loss, metric: Metric) -> None:
        """Compiles the model.

        Parameters
        ----------
        loss_fn : Loss
            Loss function to be used to compute losses and gradients.
        metric : Metric
            Metric to be used to evaluate the model.
        """
        self.loss_fn = loss_fn
        self.metric = metric
        self.compiled = True

    @abstractmethod
    def train(
        self,
        x: Tensor,
        y: Tensor,
        epochs: int = 100,
        batch_size: int | None = None,
        verbose: str = "reduced",
        val_data: tuple[Tensor, Tensor] | None = None,
        reset_params: bool = True,
    ) -> list[float]:
        """Trains the model."""

    @abstractmethod
    def evaluate(self, x: Tensor, y: Tensor) -> tuple[float | None, float | None]:
        """Evaluates the model."""


@dataclass(init=False, repr=False)
class Sequential(Model):
    """Feed forward neural network model."""

    def __init__(self, layers: list[Module]) -> None:
        """Feed forward neural network model.

        Parameters
        ----------
        layers : list[Module]
            List of modules that make up the model.
        """
        super().__init__()
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        """Computes a prediction.

        Parameters
        ----------
        x : Tensor
            Tensor of input values (features).

        Returns
        -------
        Tensor
            Tensor of predicted values.
        """
        tu.check_dims(x, len(self.input_shape))
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        if not self.compiled:
            return "Sequential. Compile model for more information about its layers."
        string = (
            f'{"layer_type":15s} | {"input_shape":15s} | {"weight_shape":15s} | '
            + f'{"bias_shape":15s} | {"output_shape":15s} | {"parameters":15s}\n\n'
        )
        sum_params = 0
        for layer in self.layers:
            string += layer.__repr__() + "\n"
            sum_params += layer.get_parameter_count()
        string += f"\ntotal trainable parameters {sum_params}"
        return string

    def compile(
        self,
        optimizer: Optimizer,
        loss_fn: Loss,
        metric: Metric,
    ) -> None:
        """Compiles the model.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer algorithm to be used to update parameters.
        loss_fn : Loss
            Loss function to be used to compute losses and gradients.
        metric : Metric
            Metric to be used to evaluate the model.

        Raises
        ------
        ModelCompilationError
            If the model has already been compiled.
        """
        if self.compiled:
            raise ModelCompilationError(
                "Model is already compiled. Initialize new model."
            )
        super().compile(loss_fn, metric)

        x = tu.ones((1, *self.layers[0].input_shape))

        for i, layer in enumerate(self.layers):
            layer.x = x
            if isinstance(layer, ParamModule):
                layer.compile(optimizer)

                # Normalization functions
                if layer.norm_name is not None:
                    norm = normalizations.NORMALIZATIONS[layer.norm_name]
                    self.layers.insert(i + 1, norm())

                # Activation functions
                if layer.act_fn_name is not None:
                    act_fn = activations.ACTIVATIONS[layer.act_fn_name]
                    index = 1 if layer.norm_name is None else 2
                    self.layers.insert(i + index, act_fn())
            else:
                layer.compile()
            x = layer(x)

        self.input_shape = self.layers[0].x.shape
        self.output_shape = self.layers[-1].y.shape
        self.compiled = True

    def train(
        self,
        x: Tensor,
        y: Tensor,
        epochs: int = 100,
        batch_size: int | None = None,
        verbose: str = "reduced",
        val_data: tuple[Tensor, Tensor] | None = None,
        reset_params: bool = True,
    ) -> tuple[list[float], list[float]]:
        """Trains the model using samples and targets.

        Parameters
        ----------
        x : Tensor
            Tensor of input values (features).
        y : Tensor
            Tensor of target values.
        epochs : int, optional
            Number of training iterations, by default 100.
        batch_size : int | None, optional
            Number of training samples used per epoch, by default None.
            If None, all samples are used.
        verbose : str, optional
            Whether to print out intermediate results while training, by default "reduced".
        val_data : tuple[Tensor, Tensor] | None, optional
            Data used for validation during training, by default None.
        reset_params : bool, optional
            Whether to reset grads after training. Improves memory usage, by default True.

        Returns
        -------
        list[float]
            List of loss values for each epoch.
        list[float] | None
            List of validation loss values for each epoch, if validation data is provided.

        Raises
        -------
        ModelCompilationError
            If the model has not been compiled yet.
        """
        if not self.compiled or not self.loss_fn:
            raise ModelCompilationError("Model not compiled yet.")
        tu.check_dims(x, len(self.input_shape))
        tu.check_dims(y, len(self.output_shape))
        train_loss_history = []
        val_loss_history = []

        for epoch in range(0, epochs + 1):
            start = time.time()
            x_train, y_train = tu.shuffle(x, y, batch_size)

            for layer in self.layers:
                layer.training = True

            # forward pass
            predictions = self(x_train)

            # compute loss
            train_loss = self.loss_fn(predictions, y_train)
            train_loss_history.append(train_loss)

            # backward pass
            y_grad = self.loss_fn.backward().data
            layers_reversed = self.layers.copy()
            layers_reversed.reverse()
            for layer in layers_reversed:
                y_grad = layer.backward(y_grad)

            for layer in self.layers:
                layer.training = False

            # validation
            val_loss = None
            if val_data is not None:
                x_val, y_val = val_data
                val_predictions = self(x_val)
                val_loss = self.loss_fn(val_predictions, y_val)
                val_loss_history.append(val_loss)

            end = time.time()
            step = round((end - start) * 1000.0, 2)
            log_training_progress(verbose, epoch, epochs, step, train_loss, val_loss)

        # reset parameters to improve memory efficiency
        if reset_params:
            self.__reset_params()

        return train_loss_history, val_loss_history

    def evaluate(self, x: Tensor, y: Tensor) -> tuple[float, float]:
        """Evaluates the model using a defined metric.

        Parameters
        ----------
        x : Tensor
            Tensor of input values (features).
        y : Tensor
            Tensor of target values.

        Returns
        ----------
        float
            Loss value.
        float
            Metric score.

        Raises
        ----------
        ModelCompilationError
            If the model has not been compiled yet.
        """
        if self.metric is None or self.loss_fn is None:
            raise ModelCompilationError("Model not compiled yet.")
        tu.check_dims(x, len(self.input_shape))
        tu.check_dims(y, len(self.output_shape))
        predictions = self(x)
        loss = self.loss_fn(predictions, y)
        score = self.metric(predictions, y)
        return loss, score

    def __reset_params(self):
        for layer in self.layers:
            layer.x.reset_params(reset_data=True)
            layer.y.reset_params(reset_data=True)
            if not layer.parameters:
                continue
            for parameter in layer.parameters:
                parameter.reset_params()
