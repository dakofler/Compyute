"""Neural network models module"""

import time
from typing import Callable
import pickle

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, ArrayLike
from walnut.nn.dataloaders import DataLoader
from walnut.nn.losses import Loss
from walnut.nn.module import Module
from walnut.nn.optimizers import Optimizer
from walnut.nn.lr_schedulers import LrScheduler
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
    train_score: float,
    val_loss: float | None,
    val_score: float | None,
) -> None:
    """Outputs information each epoch about intermediate model training results."""
    line = f"\rStep {n_steps:5d}/{n_steps} | {step_time:8.2f} ms/step | "
    line += f"train_loss {train_loss:8.4f} | train_score {train_score:6.2f}"

    if val_loss is not None:
        line += f" | val_loss {val_loss:8.4f} | val_score {val_score:6.2f}"

    print(line)


class Model(Module):
    """Neural network model base class."""

    def __init__(self) -> None:
        """Neural network model base class."""
        super().__init__()
        self.optimizer: Optimizer | None = None
        self.lr_scheduler: LrScheduler | None = None
        self.loss_fn: Loss | None = None
        self.metric_fn: Callable[[Tensor, Tensor], Tensor] | None = None
        self._compiled = False

    @property
    def compiled(self) -> bool:
        """Whether the model has been compiled yet."""
        return self._compiled

    def compile(
        self,
        optimizer: Optimizer,
        loss_fn: Loss,
        metric_fn: Callable[[Tensor, Tensor], Tensor],
        lr_scheduler: LrScheduler | None = None,
    ) -> None:
        """Compiles the model.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer algorithm to be used to update parameters.
        loss_fn : Loss
            Loss function used to compute losses and their gradients.
        metric_fn : Callable[[Tensor, Tensor], float]
            Metric function used to evaluate the model's performance.
        lr_scheduler : LrScheduler | None, optional
            Learning rate scheduler to update the optimizers learning rate, by default None.
        """
        self._compiled = True
        self.optimizer = optimizer
        optimizer.parameters = self.parameters()

        if lr_scheduler:
            self.lr_scheduler = lr_scheduler
            lr_scheduler.optimizer = optimizer

        self.loss_fn = loss_fn
        self.metric_fn = metric_fn

    def train(
        self,
        x: Tensor,
        y: Tensor,
        epochs: int = 100,
        verbose: bool = True,
        val_data: tuple[Tensor, Tensor] | None = None,
        batch_size: int = 1,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Trains the model using samples and targets.

        Parameters
        ----------
        x : Tensor
            Feature tensor.
        y : Tensor
            Label tensor.
        epochs : int, optional
            Number of training iterations, by default 100.
        batch_size : int, optional
            Number of inputs processed in parallel, by default 1.
        verbose : bool, optional
            Whether the model reports intermediate results during training, by default True.
        val_dataloader : DataLoader, optional
            Data loader for vaidation data., by default None.

        Returns
        -------
        list[float]
            Training loss history.
        list[float]
            Training score history.
        list[float] | None
            Validation loss history, if validation data is provided.
        list[float] | None
            Validation score history, if validation data is provided.

        Raises
        -------
        ModelCompilationError
            If the model has not been compiled yet.
        """
        if not self.compiled:
            raise ModelCompilationError("Model has not been compiled yet.")

        # create dataloaders
        train_dataloader = DataLoader(x, y, batch_size)
        val_dataloader = DataLoader(*val_data, batch_size) if val_data else None

        train_losses, train_scores = [], []
        val_losses, val_scores = [], []
        step_times = []

        for epoch in range(1, epochs + 1):
            if verbose:
                print(f"Epoch {epoch}/{epochs}")

            # training
            self.training = True
            n_train_steps = len(train_dataloader)

            for step, train_batch in enumerate(train_dataloader(drop_remaining=True)):
                start = time.time()

                # prepare data
                x_train_b, y_train_b = train_batch
                x_train_b.to_device(self.device)
                y_train_b.to_device(self.device)

                # forward pass
                train_loss, train_score = self._get_loss_and_score(
                    self(x_train_b), y_train_b
                )
                train_losses.append(train_loss)
                train_scores.append(train_score)

                # backward pass
                self.optimizer.reset_grads()
                dy = self.loss_fn.backward()
                self.backward(dy)

                # update model parameters
                self.optimizer.step()

                step_time = round((time.time() - start) * 1000.0, 2)
                step_times.append(step_time)
                if verbose:
                    log_step((step + 1), n_train_steps, step_time)

            avg_train_loss = sum(train_losses[-n_train_steps:]) / n_train_steps
            avg_train_score = sum(train_scores[-n_train_steps:]) / n_train_steps
            avg_step_time = sum(step_times[-n_train_steps:]) / n_train_steps
            self.training = False

            # update learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # validation
            avg_val_loss = avg_val_score = None
            if val_dataloader is not None:
                n_val_steps = max(1, len(val_dataloader))

                for val_batch in val_dataloader(shuffle=False, drop_remaining=True):
                    x_val_b, y_val_b = val_batch
                    x_val_b.to_device(self.device)
                    y_val_b.to_device(self.device)

                    val_loss, val_score = self._get_loss_and_score(
                        self(x_val_b), y_val_b
                    )
                    val_losses.append(val_loss)
                    val_scores.append(val_score)
                avg_val_loss = sum(val_losses[-n_val_steps:]) / n_val_steps
                avg_val_score = sum(val_scores[-n_val_steps:]) / n_val_steps

            if verbose:
                log_epoch(
                    n_train_steps,
                    avg_step_time,
                    avg_train_loss,
                    avg_train_score,
                    avg_val_loss,
                    avg_val_score,
                )

        if not self.remember:
            self.reset()

        return train_losses, train_scores, val_losses, val_scores

    def evaluate(
        self, x: Tensor, y: Tensor, batch_size: int = 1
    ) -> tuple[float, float]:
        """Evaluates the model using a defined metric.

        Parameters
        ----------
        x : Tensor
            Feature tensor.
        y : Tensor
            Label tensor.
        batch_size : int, optional
            Number of inputs processed in parallel, by default 1.

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

        y_pred = self.predict(x, batch_size)
        y.to_device(self.device)
        return self._get_loss_and_score(y_pred, y)

    def predict(self, x: Tensor, batch_size: int = 1) -> Tensor:
        """Returns the models predictions for a given input.

        Parameters
        ----------
        x : Tensor
            Feature tensor.
        batch_size : int, optional
            Number of inputs processed in parallel, by default 1.

        Returns
        -------
        Tensor
            Predictions.
        """

        dataloader = DataLoader(x, None, batch_size)
        outputs = []

        for batch in dataloader(shuffle=False):
            x, _ = batch
            x.to_device(self.device)
            outputs.append(self(x))
        if not self.remember:
            self.reset()

        return tu.concatenate(outputs, axis=0)

    def _get_loss_and_score(self, y_pred, y_true) -> tuple[float, float]:
        loss = self.loss_fn(y_pred, y_true).item()
        score = self.metric_fn(y_pred, y_true).item()
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
    model.loss_fn.backward = None
    model.reset()

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
