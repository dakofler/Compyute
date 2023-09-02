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


def log_step(
    step: int,
    n_steps: int,
    step_time: float,
    step_train_loss: float,
    step_val_loss: float | None,
) -> None:
    """Outputs information each step about intermediate model training results."""
    line = f"\rStep {step:5d}/{n_steps} | {step_time:8.2f} ms/step | loss {step_train_loss:8.4f}"
    if step_val_loss is not None:
        line += f" | val_loss {step_val_loss:8.4f}"
    print(line, end="")


def log_epoch(
    n_steps: int,
    step_time: float,
    train_loss: float,
    val_loss: float | None,
) -> None:
    """Outputs information each epoch about intermediate model training results."""
    line = f"\rStep {n_steps:5d}/{n_steps} | {step_time:8.2f} ms/step | loss {train_loss:8.4f}"
    if val_loss is not None:
        line += f" | val_loss {val_loss:8.4f}"
    print(line)


def get_batches(x: Tensor, y: Tensor, batch_size: int | None) -> tuple[Tensor, Tensor]:
    """Generates batches for training a model.

    Parameters
    ----------
    x : Tensor
        Tensor of input values (features).
    y : Tensor
        Tensor of target values (labels).
    batch_size : int | None, optional
        Batch size. If None, batch_size=1 is used.

    Returns
    -------
    tuple[Tensor, Tensor]
        Batched input and target values.
    """
    x_shuffled, y_shuffled = tu.shuffle(x, y)
    batch_size = min(x.len, batch_size) if isinstance(batch_size, int) else x.len
    n = x_shuffled.len // batch_size * batch_size
    x_batches = x_shuffled[:n].reshape((-1, batch_size, *x_shuffled.shape[1:]))
    y_batches = y_shuffled[:n].reshape((-1, batch_size, *y_shuffled.shape[1:]))
    return x_batches, y_batches


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
        verbose: bool = True,
        val_data: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Trains the model using samples and targets.

        Parameters
        ----------
        x : Tensor
            Tensor of input values (features).
        y : Tensor
            Tensor of target values (labels).
        epochs : int, optional
            Number of training iterations, by default 100.
        batch_size : int | None, optional
            Number of training samples used per epoch, by default None.
            If None, all samples are used.
        verbose : bool, optional
            Whether the model reports intermediate results during training, by default True.
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
            epoch_train_loss = avg_step_time = 0.0
            epoch_val_loss = 0.0 if val_data is not None else None

            if verbose:
                print(f"Epoch {epoch}/{epochs}")
            x_batched, y_batched = get_batches(x, y, batch_size)
            n_steps = x_batched.shape[0]

            for step in range(n_steps):
                self.training_mode()
                start = time.time()

                x_train = x_batched[step]
                y_train = y_batched[step]

                # forward pass
                preds = self(x_train)

                # compute loss
                train_loss = self.loss_fn(preds, y_train).item()
                epoch_train_loss += train_loss / n_steps

                # backward pass
                self.reset_grads()
                y_grad = self.loss_fn.backward()
                self.backward(y_grad)

                # update parameters
                t = (epoch - 1) * n_steps + step + 1
                self.optimizer.step(t)

                step_time = round((time.time() - start) * 1000.0, 2)
                avg_step_time += step_time / n_steps

                self.eval_mode()

                # validation
                val_loss = None
                if val_data is not None:
                    x_val_batched, y_val_batched = get_batches(*val_data, batch_size)
                    val_loss = 0.0
                    n_val_steps = x_val_batched.shape[0]

                    for v_step in range(n_val_steps):
                        x_val = x_val_batched[v_step]
                        y_val = y_val_batched[v_step]
                        val_preds = self(x_val)
                        val_loss += self.loss_fn(val_preds, y_val).item() / n_val_steps

                    epoch_val_loss += val_loss / n_steps

                if verbose:
                    log_step((step + 1), n_steps, step_time, train_loss, val_loss)

            if verbose:
                log_epoch(n_steps, avg_step_time, epoch_train_loss, epoch_val_loss)

            train_loss_history.append(epoch_train_loss)
            if val_data is not None:
                val_loss_history.append(epoch_val_loss)

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
