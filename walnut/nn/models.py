"""Neural network models module"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

from walnut import tensor
from walnut.tensor import Tensor, ShapeLike
from walnut.nn.losses import Loss
from walnut.nn.metrics import Metric
from walnut.nn.optimizers import Optimizer
from walnut.nn.layers import activations
from walnut.nn.layers import normalizations
from walnut.nn.layers.parameter import Layer, ParamLayer


class ModelCompilationError(Exception):
    """Error with the compiling of the model."""


@dataclass()
class Model(ABC):
    """Neural network model base class."""

    loss_fn: Loss | None = None
    metric: Metric | None = None
    compiled: bool = False
    input_shape: ShapeLike | None = None
    output_shape: ShapeLike | None = None

    def __call__(self, X: Tensor, Y: Tensor | None = None) -> None:
        self.__check_dims(X, Y)

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
        X: Tensor,
        Y: Tensor,
        epochs: int = 100,
        batch_size: int | None = None,
        verbose: bool = True,
        val_data: tuple[Tensor, Tensor] | None = None,
    ) -> list[float]:
        """Trains the model."""

    @abstractmethod
    def forward(self):
        """Performs a forward pass."""

    @abstractmethod
    def backward(self):
        """Performs a backward pass."""

    @abstractmethod
    def evaluate(self, X: Tensor, Y: Tensor) -> tuple[float | None, float | None]:
        """Evaluates the model."""

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

    def __call__(self, X: Tensor, mode: str = "eval") -> Tensor:
        """Computes a prediction.

        Parameters
        ----------
        X : Tensor
            Tensor of input values (features).
        mode : str, optional
            Defines the model mode used for the pass.

        Returns
        -------
        Tensor
            Tensor of predicted values.
        """
        super().__call__(X)
        self.layers[0].x.data = X.data.copy()
        self.forward(mode)
        return Tensor(self.layers[-1].y.data)

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
        for i, layer in enumerate(self.layers):
            if i > 0:
                layer.x.data = self.layers[i - 1].y.data
            if isinstance(layer, ParamLayer):
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
            layer.forward()
        self.input_shape = self.layers[0].x.shape
        self.output_shape = self.layers[-1].y.shape
        self.compiled = True

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

        Raises
        -------
        ModelCompilationError
            If the model has not been compiled yet.
        """
        if not self.compiled:
            raise ModelCompilationError("Model not compiled yet.")
        history = []
        val_loss = None
        if self.loss_fn is None:
            return history

        for epoch in range(1, epochs + 1):
            start = time.time()
            x_train, y_train = tensor.shuffle(X, Y, batch_size)

            # forward pass
            prediction = self(x_train, mode="train")

            # compute loss
            loss = self.loss_fn(prediction, y_train)

            # backward pass
            loss_grad = self.loss_fn.backward().data.copy()
            self.layers[-1].y.grad = loss_grad
            self.backward()

            # validation
            if val_data is not None:
                x_val, y_val = val_data
                val_predictions = self(x_val)
                val_loss = self.loss_fn(val_predictions, y_val)

            end = time.time()
            step = round((end - start) * 1000.0, 2)
            if verbose and loss is not None:
                self.log(epoch, epochs, step, loss, val_loss)
            history.append(loss)

        return history

    def evaluate(self, X: Tensor, Y: Tensor) -> tuple[float, float]:
        """Evaluates the model using a defined metric.

        Parameters
        ----------
        X : Tensor
            Tensor of input values (features).
        Y : Tensor
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
        predictions = self(X)
        loss = self.loss_fn(predictions, Y)
        score = self.metric(predictions, Y)
        return loss, score

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

    def forward(self, mode: str = "eval") -> None:
        for i, layer in enumerate(self.layers):
            if i > 0:
                layer.x.data = self.layers[i - 1].y.data
            layer.forward(mode)

    def backward(self):
        layers_reversed = self.layers.copy()
        layers_reversed.reverse()
        for i, layer in enumerate(layers_reversed):
            if i > 0:
                layer.y.grad = layers_reversed[i - 1].x.grad
            layer.backward()
