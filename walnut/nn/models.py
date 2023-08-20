"""Neural network models module"""

import time

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NumpyArray
from walnut.nn.logging import log_training_progress
from walnut.nn.losses import Loss
from walnut.nn.metrics import Metric
from walnut.nn.module import Module
from walnut.nn.optimizers import Optimizer
from walnut.nn.layers import Linear, Tanh, Recurrent


__all__ = ["Model", "Sequential", "RNN"]


class ModelCompilationError(Exception):
    """Error if the model has not been compiled yet."""


class MissingLayersError(Exception):
    """Error when the list of layers is empty."""


class Model(Module):
    """Neural network model base class."""

    def __init__(self) -> None:
        super().__init__()
        self.optimizer: Optimizer | None = None
        self.loss_fn: Loss | None = None
        self.metric: Metric | None = None

    def compile(self, optimizer: Optimizer, loss_fn: Loss, metric: Metric) -> None:
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
        MissingLayersError
            If the model does not contain any layers.
        """
        if len(self.layers) == 0:
            raise MissingLayersError("Module does not contain any layers.")

        self.optimizer = optimizer
        optimizer.parameters = self.get_parameters()
        self.loss_fn = loss_fn
        self.metric = metric

    def train(
        self,
        x: Tensor,
        y: Tensor,
        epochs: int = 100,
        batch_size: int | None = None,
        verbose: int | None = 10,
        val_data: tuple[Tensor, Tensor] | None = None,
        reset_grads: bool = True,
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
        verbose : int | None, optional
            How often to report intermediate results.
            If None, no results are reported. If 0 all results are reported, by default 10.
        val_data : tuple[Tensor, Tensor] | None, optional
            Data used for validation during training, by default None.
        reset_grads : bool, optional
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
        if not self.loss_fn or not self.optimizer:
            raise ModelCompilationError("Model is not compiled yet.")

        train_loss_history, val_loss_history = [], []

        for epoch in range(1, epochs + 1):
            start = time.time()
            x_train, y_train = tu.shuffle(x, y, batch_size)
            self.training_mode()

            # forward pass
            predictions = self(x_train)

            # compute loss
            train_loss = self.loss_fn(predictions, y_train).item()
            train_loss_history.append(train_loss)

            # backward pass
            self.reset_grads()
            y_grad = self.loss_fn.backward()
            self.backward(y_grad)

            # update parameters
            self.optimizer.step()

            self.eval_mode()

            # validation
            val_loss = None
            if val_data is not None:
                x_val, y_val = val_data
                val_predictions = self(x_val)
                val_loss = self.loss_fn(val_predictions, y_val).item()
                val_loss_history.append(val_loss)

            end = time.time()

            if verbose is not None and (
                verbose == 0 or epoch == 1 or epoch % (epochs // verbose) == 0
            ):
                step = round((end - start) * 1000.0, 2)
                log_training_progress(epoch, epochs, step, train_loss, val_loss)

        # reset parameters to improve memory efficiency
        if reset_grads:
            self.reset_grads()

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
        if not self.metric or not self.loss_fn:
            raise ModelCompilationError("Model not compiled yet.")

        predictions = self(x)
        loss = self.loss_fn(predictions, y).item()
        score = self.metric(predictions, y)
        return loss, score


class Sequential(Model):
    """Feed forward neural network model."""

    def __init__(self, layers: list[Module]) -> None:
        """Feed forward neural network model.

        Parameters
        ----------
        layers : list[Module]
            List of layers used in the model. These layers are processed sequentially.
        """
        super().__init__()
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                layers_reversed = self.layers.copy()
                layers_reversed.reverse()

                for layer in layers_reversed:
                    y_grad = layer.backward(y_grad)
                return y_grad

            self.backward = backward

        return x


class RNN(Model):
    """Recurrent neural network model."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        activation: str = "tanh",
        num_layers: int = 1,
        use_bias: bool = True,
    ) -> None:
        """Recurrent neural network model.

        Parameters
        ----------
        in_channels : int
            Number of input features.
        hidden_channels: int
            Number of hidden features.
        activation: Module, optional
            Activation function to be used in the hidden layer, by default Tanh().
        num_layers: int, optional
            Number of recurrent layers in the model, by default 1.
        use_bias : bool, optional
            Whether to use bias values, by default True.
        """
        super().__init__()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    Linear(in_channels, hidden_channels, use_bias=use_bias)
                )
            else:
                self.layers.append(
                    Linear(hidden_channels, hidden_channels, use_bias=use_bias)
                )
            self.layers.append(
                Recurrent(hidden_channels, activation, use_bias=use_bias)
            )

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        if self.training:

            def backward(y_grad: NumpyArray) -> NumpyArray:
                layers_reversed = self.layers.copy()
                layers_reversed.reverse()

                for layer in layers_reversed:
                    y_grad = layer.backward(y_grad)
                return y_grad

            self.backward = backward

        return x
