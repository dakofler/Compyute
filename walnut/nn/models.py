"""Neural network models module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from typing import Callable

from walnut import tensor
from walnut.tensor import Tensor
from walnut.nn.losses import Loss
from walnut.nn.optimizers import Optimizer
from walnut.nn.layers.parameter import Layer, ParamLayer


@dataclass()
class Model(ABC):
    """Neural network model base class."""

    optimizer: Optimizer | None = None
    loss_fn: Loss | None = None
    metric: Callable[[Tensor, Tensor], float] | None = None
    compiled: bool = False
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None

    def __call__(self, X: Tensor, Y: Tensor | None = None) -> None:
        self.__check_dims(X, Y)

    def compile(
        self,
        loss_fn: Loss,
        metric: Callable[[Tensor, Tensor], float],
    ) -> None:
        """Compiles the model.

        Parameters
        ----------
        loss_fn : losses.Loss
            Loss function to be used to compute losses and gradients.
        metric : Callable[[Tensor, Tensor], float]
            Metric to be used to evaluate the model.
        """
        self.loss_fn = loss_fn
        self.metric = metric
        self.compiled = True

    @abstractmethod
    def train(
        self,
        X: Tensor,
        Y: Tensor,
        epochs: int = 100,
        batch_size: int | None = None,
        verbose: bool = True,
        val_data: tuple[Tensor, Tensor] | None = None,
    ) -> list[float]:
        """Trains the model."""

    @abstractmethod
    def evaluate(self, X: Tensor, Y: Tensor) -> tuple[float | None, float | None]:
        """Evaluates the model."""

    @abstractmethod
    def predict(self, X: Tensor) -> Tensor:
        """Makes a prediction."""

    def log(
        self,
        epoch: int,
        epochs: int,
        step: float,
        loss: float,
        val_loss: float | None = None,
    ):
        """Logs information while training the model

        Parameters
        ----------
        epoch : int
            Current traning epoch.
        epochs : int
            Total number of epochs.
        step : float
            Computation time for the current epoch.
        loss : float
            Current training loss.
        val_loss : float | None, optional
            Current validation loss, by default None.
        """

        def __log_line():
            line = (
                f"epoch {epoch:5d}/{epochs:5d} | step {step:.2f} ms | loss {loss:.4f}"
            )
            if val_loss is not None:
                line += f" | val_loss {val_loss:.4f}"
            print(line)

        if epochs < 10:
            __log_line()
        elif epoch % (epochs // 10) == 0:
            __log_line()

    def __check_dims(self, x: Tensor, y: Tensor | None = None):
        if not self.input_shape or not self.output_shape:
            return
        req_input_dim = len(self.input_shape)
        if x.ndim != req_input_dim:
            raise ValueError(f"Input dimension must be {req_input_dim}.")
        if y is not None:
            req_output_dim = len(self.output_shape)
            if y.ndim != req_output_dim:
                raise ValueError(f"Output dimension must be {req_output_dim}.")


@dataclass(init=False)
class Sequential(Model):
    """Feed forward neural network model."""

    def __init__(self, layers: list[Layer]) -> None:
        """Feed forward neural network model.

        Parameters
        ----------
        layers : list[Layer]
            List of layers that make up the model.
        """
        super().__init__()
        self.layers = layers

    def __call__(
        self, X: Tensor, Y: Tensor | None = None, mode: str = "eval"
    ) -> tuple[Tensor, float | None]:
        """Computes a prediction and loss.

        Parameters
        ----------
        X : Tensor
            Tensor of input values (features).
        Y : Tensor | None, optional
            Tensor of target values, by default None.

        Returns
        -------
        Tensor
            Tensor of predicted values.
        Tensor
            Loss value, if target values are provided else None.
        """
        super().__call__(X, Y)
        self.layers[0].x.data = X.data.copy()
        self.__forward(mode)
        output = Tensor(self.layers[-1].y.data)

        loss = None
        if self.loss_fn is not None and Y is not None:
            loss = self.loss_fn(output, Y)
        return output, loss

    def compile(
        self,
        optimizer: Optimizer,
        loss_fn: Loss,
        metric: Callable[[Tensor, Tensor], float],
    ) -> None:
        """Compiles the model.

        Parameters
        ----------
        optimizer : optimizers.Optimizer
            Optimizer algorithm to be used to update parameters.
        loss_fn : losses.Loss
            Loss function to be used to compute losses and gradients.
        metric : Callable[[Tensor, Tensor], float]
            Metric to be used to evaluate the model.
        """
        super().compile(loss_fn, metric)

        num_layers = len(self.layers) - 1

        # connect layers
        for i, layer in enumerate(self.layers):
            if 0 < i:
                layer.prev_layer = self.layers[i - 1]
            if i < num_layers:
                layer.next_layer = self.layers[i + 1]
            layer.compile()
            layer.forward()

            # set optimizer for parameter layers
            if isinstance(layer, ParamLayer):
                layer.optimizer = optimizer

        self.input_shape = self.layers[0].x.shape
        self.output_shape = self.layers[-1].y.shape

    def train(
        self,
        X: Tensor,
        Y: Tensor,
        epochs: int = 100,
        batch_size: int | None = None,
        verbose: bool = True,
        val_data: tuple[Tensor, Tensor] | None = None,
    ) -> list[float]:
        """Trains the model using samples and targets.

        Parameters
        ----------
        X : Tensor
            Tensor of input values (features).
        Y : Tensor
            Tensor of target values.
        epochs : int, optional
            Number of training iterations, by default 100.
        batch_size : int | None, optional
            Number of training samples used per epoch, by default None.
            If None, all samples are used.
        verbose : bool, optional
            Whether to print out intermediate results while training, by default True.
        val_data : tuple[Tensor, Tensor] | None, optional
            Data used for validation during training, by default None.

        Returns
        -------
        list[float]
            List of loss values for each epoch.
        """
        history = []
        val_loss = None

        for epoch in range(1, epochs + 1):
            start = time.time()
            x_train, y_train = tensor.shuffle(X, Y, batch_size)

            # training
            _, loss = self(x_train, y_train, mode="train")
            self.__backward()

            # validation
            if val_data is not None:
                _, val_loss = self(*val_data)

            end = time.time()
            step = round((end - start) * 1000.0, 2)
            if verbose and loss is not None:
                self.log(epoch, epochs, step, loss, val_loss)
            history.append(loss)

        return history

    def predict(self, X: Tensor) -> Tensor:
        """Applies the input to the model and returns it's predictions.

        Parameters
        ----------
        X : Tensor
            Tensor of input features.

        Returns
        -------
        Tensor
            Tensor of predicted values.
        """
        pred, _ = self(X)
        return pred

    def evaluate(self, X: Tensor, Y: Tensor) -> tuple[float | None, float | None]:
        """Evaluates the model using a defined metric.

        Parameters
        ----------
        X : Tensor
            Tensor of input values (features).
        Y : Tensor
            Tensor of target values.
        """
        if self.metric is None:
            return None, None
        outputs, loss = self(X, Y)
        score = self.metric(outputs, Y)
        return loss, score

    def summary(self, reduced: bool = False) -> None:
        """Gives an overview of the model architecture.

        Parameters
        ----------
        reduced : bool, optional
            If True, only parameter layers are shown, by default False.

        Raises
        ------
        ValueError
            If the model has not been compiled yet.
        """
        if not self.compiled:
            raise ValueError("Model has not been compiled yet.")
        print(
            f'{"layer_type":15s} | {"input_shape":15s} | {"weight_shape":15s} | '
            + f'{"bias_shape":15s} | {"output_shape":15s} | {"parameters":15s}\n'
        )
        sum_params = 0
        for layer in self.layers:
            if reduced and not isinstance(layer, ParamLayer):
                continue
            print(layer)
            sum_params += layer.get_parameter_count()
        print(f"\ntotal trainable parameters {sum_params}")

    def __forward(self, mode: str = "eval") -> None:
        for layer in self.layers:
            layer.forward(mode)

    def __backward(self):
        if self.loss_fn is None:
            return
        self.layers[-1].y.grad = self.loss_fn.backward().data.copy()
        layers_reversed = self.layers.copy()
        layers_reversed.reverse()
        for layer in layers_reversed:
            layer.backward()
