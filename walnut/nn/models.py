"""Neural network models module"""

import time
from typing import Callable

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NpArrayLike
from walnut.nn.losses import Loss
from walnut.nn.module import Module
from walnut.nn.optimizers import Optimizer
from walnut.nn.containers import Container, SequentialContainer


__all__ = ["Model", "Sequential"]


class ModelCompilationError(Exception):
    """Error if the model has not been compiled yet."""


def log_training_progress(
    epoch: int,
    epochs: int,
    time_step: float,
    training_loss: float,
    validation_loss: float | None,
) -> None:
    """Prints out information about intermediate model training results."""
    line = f"epoch {epoch:5d}/{epochs:5d} | step {time_step:9.2f} ms | loss {training_loss:8.4f}"
    if validation_loss is not None:
        line += f" | val_loss {validation_loss:8.4f}"
    print(line)


class Model(Container):
    """Neural network model base class."""

    def __init__(self) -> None:
        """Neural network model base class."""
        super().__init__()
        self.optimizer: Optimizer | None = None
        self.loss_fn: Loss | None = None
        self.metric: Callable[[Tensor, Tensor], float] = lambda x, y: 0.0

    def compile(
        self,
        optimizer: Optimizer,
        loss_fn: Loss,
        metric: Callable[[Tensor, Tensor], float],
    ) -> None:
        """Compiles the model.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer algorithm to be used to update parameters.
        loss_fn : Loss
            Loss function used to compute losses and their gradients.
        metric : Callable[[Tensor, Tensor], float]
            Metric function used to evaluate the model's performance.
        """
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
            self.optimizer.step(epoch)

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
                verbose == 0 or epoch == 1 or epoch % max(1, epochs // verbose) == 0
            ):
                step = round((end - start) * 1000.0, 2)
                log_training_progress(epoch, epochs, step, train_loss, val_loss)

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
        self.layers = [SequentialContainer(layers)]

    def __call__(self, x: Tensor) -> Tensor:
        y = self.layers[0](x)

        if self.training:

            def backward(y_grad: NpArrayLike) -> NpArrayLike:
                self.set_y_grad(y_grad)
                return self.layers[0].backward(y_grad)

            self.backward = backward

        self.set_y(y)
        return y
