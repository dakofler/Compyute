"""Model trainer."""

from typing import Literal, Optional

from ...tensors import Tensor
from ...typing import ScalarLike
from ..losses import _LossLike, get_loss_function
from ..metrics import _MetricLike, get_metric_function
from ..modules.module import Module
from ..optimizers import _OptimizerLike, get_optimizer
from ..utils import Dataloader
from .callbacks import Callback

__all__ = ["Trainer"]


class Trainer:
    """Trainer utility used to train neural network models.

    Parameters
    ----------
    model : Module
        Model to be trained.
    optimizer : _OptimizerLike
        Optimizer algorithm used to update model parameters based on gradients.
        See :ref:`optimizers` for more details.
    loss : _LossLike
        Loss function used to quantify the model performance.
        See :ref:`losses` for more details.
    metric : _MetricLike, optional
        Metric function used to evaluate the model. Defaults to ``None``.
        See :ref:`metrics` for more details.
    callbacks : list[Callback], optional
        Callback functions that are executing each training epoch and step. Defaults to ``None``.
        See :ref:`callbacks` for more details.
    """

    cache: dict = {"abort": False, "t": 1}

    def __init__(
        self,
        model: Module,
        optimizer: _OptimizerLike,
        loss: _LossLike,
        metric: Optional[_MetricLike] = None,
        callbacks: Optional[list[Callback]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = get_optimizer(optimizer)
        self.optimizer.parameters = model.get_parameters()
        self.loss = get_loss_function(loss)
        self.metric = None if metric is None else get_metric_function(metric)
        self.metric_name = (
            None if metric is None else self.metric.__class__.__name__.lower()
        )
        self.callbacks = callbacks

    def train(
        self,
        x_train: Tensor,
        y_train: Tensor,
        epochs: int = 100,
        val_data: Optional[tuple[Tensor, Tensor]] = None,
        batch_size: int = 32,
    ) -> None:
        """Trains the model using samples and targets.

        Parameters
        ----------
        X_train : Tensor
            Input tensor.
        y_train : Tensor
            Target tensor.
        epochs : int, optional
            Number of training iterations. Defaults to ``100``.
        val_data : tuple[Tensor, Tensor], optional
            Data used for the validaton every epoch. Defaults to ``None``.
        batch_size : int, optional
            Number of inputs processed in parallel. Defaults to ``32``.
            If ``-1``, all samples are used.
        """
        batch_size = len(x_train) if batch_size == -1 else batch_size
        train_dataloader = Dataloader(x_train, y_train, batch_size, self.model.device)
        self.cache["epochs"] = epochs
        self.cache["train_steps"] = len(train_dataloader)
        self._callback("start")

        for t in range(1, epochs + 1):
            self.cache["t"] = t
            self._callback("epoch_start")

            # training
            with self.model.train():
                for s, batch in enumerate(train_dataloader(), 1):
                    self.cache["step"] = s
                    self._callback("step_start")
                    self.cache["lr"] = self.optimizer.lr
                    self._train_step(batch)
                    self._callback("step_end")

            # validation
            if val_data:
                loss, score = self.evaluate_model(*val_data, batch_size=batch_size)
                self.cache["val_loss"] = loss
                if self.metric is not None:
                    self.cache[f"val_{self.metric_name}_score"] = score

            self._callback("epoch_end")
            if self.cache["abort"]:
                break

        self._callback("end")
        self.model.clean()

    def evaluate_model(
        self, x: Tensor, y: Tensor, batch_size: int = 32
    ) -> tuple[ScalarLike, Optional[ScalarLike]]:
        """Evaluates the model using a defined metric.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        y : Tensor
            Target tensor.
        batch_size : int, optional
            Number of inputs processed in parallel. Defaults to ``32``.

        Returns
        ----------
        _ScalarLike
            Loss value.
        _ScalarLike, optional
            Metric score.
        """
        dataloader = Dataloader(
            x,
            y,
            batch_size=batch_size,
            device=self.model.device,
            shuffle_data=False,
        )

        losses, scores = [], []

        # compute loss/score for each batch to save memory
        for x_batch, y_batch in dataloader():
            y_pred = self.model(x_batch)
            losses.append(self.loss(y_pred, y_batch).item())
            if self.metric is not None:
                scores.append(self.metric(y_pred, y_batch).item())

        loss = sum(losses) / len(losses)
        if self.metric is not None:
            return loss, sum(scores) / len(scores)
        return loss, None

    def _callback(
        self,
        on: Literal[
            "start", "step_start", "step_end", "epoch_start", "epoch_end", "end"
        ],
    ) -> None:
        if self.callbacks is None:
            return
        for callback in self.callbacks:
            {
                "start": callback.on_start,
                "step_start": callback.on_step_start,
                "step_end": callback.on_step_end,
                "epoch_start": callback.on_epoch_start,
                "epoch_end": callback.on_epoch_end,
                "end": callback.on_training_end,
            }[on](self.cache)

    def _train_step(self, batch: tuple[Tensor, Tensor]) -> None:
        # prepare data
        x_batch, y_batch = batch

        # forward pass
        y_pred = self.model(x_batch)

        # compute loss and metrics
        self.cache["loss"] = self.loss(y_pred, y_batch).item()
        if self.metric is not None:
            key = f"{self.metric_name}_score"
            self.cache[key] = self.metric(y_pred, y_batch).item()

        # backward pass
        self.optimizer.reset_grads()
        self.model.backward(self.loss.backward())

        # update parameters
        self.optimizer.step()
