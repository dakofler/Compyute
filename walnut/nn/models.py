"""Neural network models module"""

import time
from typing import Callable
import pickle

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ArrayLike
from walnut.nn.losses import Loss
from walnut.nn.module import Module
from walnut.nn.optimizers import Optimizer
from walnut.nn.containers import SequentialContainer


__all__ = ["Model", "Sequential", "save_model", "load_model"]


class ModelCompilationError(Exception):
    """Error if the model has not been compiled yet."""


def log_step(step: int, n_steps: int, step_time: float) -> None:
    """Outputs information each step about intermediate model training results."""
    eta = (n_steps - step) * step_time / 1000.0
    line = f"\rStep {step:5d}/{n_steps} | ETA: {eta:6.1f} s"
    print(line, end="")


def log_epoch(
    n_steps: int,
    step_time: float,
    train_loss: float,
    val_loss: float | None,
) -> None:
    """Outputs information each epoch about intermediate model training results."""
    line = f"\rStep {n_steps:5d}/{n_steps} | {step_time:8.2f} ms/step | train_loss {train_loss:8.4f}"
    if val_loss is not None:
        line += f" | val_loss {val_loss:8.4f}"
    print(line)


def get_batches(
    x: Tensor,
    y: Tensor | None = None,
    batch_size: int | None = None,
    shuffle: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Generates batches of data."""

    if isinstance(batch_size, int) and (batch_size > x.len or batch_size == 0):
        raise ValueError(f"Invalid batch_size {batch_size}.")

    if shuffle:
        x, idx = tu.shuffle(x)
        y = y[idx]

    batch_size = x.len if batch_size is None else batch_size
    clip = x.len // batch_size * batch_size
    x_batches = x[:clip].reshape((-1, batch_size, *x.shape[1:]))

    if y is None:
        return x_batches, None
    return x_batches, y[:clip].reshape((-1, batch_size, *y.shape[1:]))


class Model(Module):
    """Neural network model base class."""

    def __init__(self) -> None:
        """Neural network model base class."""
        super().__init__()
        self.optimizer: Optimizer | None = None
        self.loss_fn: Loss | None = None
        self.metric: Callable[[Tensor, Tensor], float] = lambda x, y: 0.0
        self._compiled = False

    @property
    def compiled(self) -> bool:
        """Whether the model has been compiled yet."""
        return self._compiled

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
        self._compiled = True
        self.optimizer = optimizer
        optimizer.parameters = self.parameters
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
        keep_intermediate_outputs: bool = False,
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
        keep_intermediate_outputs : bool, optional
            Whether to store output values and gradients of sub modules, by default False.
            Sub module outputs and gradients are kept in as a y-tensor.

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
        if not self.compiled:
            raise ModelCompilationError("Model has not been compiled yet.")

        self.keep_output = keep_intermediate_outputs
        train_loss_history, val_loss_history = [], []

        for epoch in range(1, epochs + 1):
            avg_train_loss = avg_val_loss = avg_step_time = 0.0

            if verbose:
                print(f"Epoch {epoch}/{epochs}")

            x_batched, y_batched = get_batches(x, y, batch_size)
            n_steps = x_batched.shape[0]

            for step in range(n_steps):
                start = time.time()
                self.training = True
                x_train = x_batched[step]
                y_train = y_batched[step]

                # forward pass
                preds = self(x_train)

                # compute loss
                train_loss = self.loss_fn(preds, y_train).item()
                avg_train_loss = avg_train_loss + train_loss / n_steps

                # backward pass
                self.optimizer.reset_grads()
                dy = self.loss_fn.backward()
                self.backward(dy)

                # update parameters
                t = (epoch - 1) * n_steps + step + 1
                self.optimizer.step(t)

                step_time = round((time.time() - start) * 1000.0, 2)
                avg_step_time = avg_step_time + step_time / n_steps
                self.training = False

                if verbose:
                    log_step((step + 1), n_steps, step_time)

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
                        val_loss = (
                            val_loss
                            + self.loss_fn(val_preds, y_val).item() / n_val_steps
                        )

                    avg_val_loss = avg_val_loss + val_loss / n_steps

                train_loss_history.append(train_loss)
                if val_data is not None:
                    val_loss_history.append(val_loss)

            if verbose:
                log_epoch(n_steps, avg_step_time, avg_train_loss, val_loss)

        self.optimizer.reset_temp_params()
        self.keep_output = False
        return train_loss_history, val_loss_history

    def evaluate(
        self, x: Tensor, y: Tensor, batch_size: int | None = None
    ) -> tuple[float, float]:
        """Evaluates the model using a defined metric.

        Parameters
        ----------
        x : Tensor
            Tensor of input values (features).
        y : Tensor
            Tensor of target values.
        batch_size : int | None, optional
            Number of samples used per call, by default None.
            If None, all samples are used.

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
        if not self.compiled:
            raise ModelCompilationError("Model has not been compiled yet.")

        predictions = self.predict(x, batch_size=batch_size)
        loss = self.loss_fn(predictions, y).item()
        score = self.metric(predictions, y)
        return loss, score

    def predict(self, x: Tensor, batch_size: int | None = None) -> Tensor:
        """Returns the models predictions for a given input.

        Parameters
        ----------
        x : Tensor
            _description_
        batch_size : int | None, optional
            Number of samples used per call, by default None.
            If None, all samples are used.

        Returns
        -------
        Tensor
            Predictions.
        """
        x_batched, _ = get_batches(x, batch_size=batch_size, shuffle=False)

        # get predictions for n * batchsize samples
        predictions = tu.concatenate([self(x) for x in x_batched], axis=0)

        # get predictions for remaining samples
        n = tu.prod(x_batched.shape[:2])
        if n < x.len:
            predictions = tu.concatenate([predictions, self(x[n:])], axis=0)

        return predictions


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
        self.sub_modules = [SequentialContainer(layers)]

    def __call__(self, x: Tensor) -> Tensor:
        y = self.sub_modules[0](x)

        if self.training:

            def backward(dy: ArrayLike) -> ArrayLike:
                self.set_dy(dy)
                return self.sub_modules[0].backward(dy)

            self.backward = backward

        self.set_y(y)
        return y


def save_model(model: Model, filename: str) -> None:
    """Saves a model as a binary file.

    Parameters
    ----------
    model : Model
        Model to be saved.
    filename : str
        Name of the file.
    """
    if not model.compiled:
        raise ModelCompilationError("Model has not been compiled yet.")

    model.to_device("cpu")
    model.optimizer.reset_grads()
    model.optimizer.reset_temp_params()
    model.loss_fn.backward = None
    model.clean()

    file = open(filename, "wb")
    pickle.dump(model, file)
    file.close()


def load_model(filename: str) -> Model:
    """Load a model from a previously saved binary file.

    Parameters
    ----------
    filename : str
        Name of the saved file.

    Returns
    -------
    Model
        Loaded model.
    """
    file = open(filename, "rb")
    obj = pickle.load(file)
    file.close()
    return obj
