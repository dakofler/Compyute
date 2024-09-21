"""Model trainer."""

from typing import Any, Literal, Optional

from ...tensors import Tensor
from ...typing import ScalarLike
from ..functional.functions import no_caching
from ..losses import Loss, _LossLike, get_loss_function
from ..metrics import Metric, _MetricLike, get_metric_function
from ..modules.module import Module
from ..optimizers import Optimizer, _OptimizerLike, get_optimizer
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

    model: Module
    optimizer: Optimizer
    loss: Loss
    metric: Optional[Metric] = None
    callbacks: Optional[list[Callback]] = None
    _metric_name: Optional[str] = None
    _cache: dict[str, Any]

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
        self.optimizer.set_parameters(model.get_parameters())
        self.loss = get_loss_function(loss)
        if metric is not None:
            self.metric = get_metric_function(metric)
            self._metric_name = self.metric.__class__.__name__.lower()
        if callbacks is not None:
            self.callbacks = callbacks

        self._cache: dict[str, Any] = {"abort": False}

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
        batch_size = batch_size if batch_size > 0 else len(x_train)
        train_dataloader = Dataloader((x_train, y_train), batch_size, self.model.device)
        self._cache["epochs"] = epochs
        self._cache["train_steps"] = len(train_dataloader)
        self._callback("start")

        try:
            for t in range(1, epochs + 1):
                self._cache["t"] = t
                self._callback("epoch_start")

                # training
                self.model.training()
                for s, batch in enumerate(train_dataloader(), 1):
                    self._cache["step"] = s
                    self._callback("step_start")
                    self._cache["lr"] = self.optimizer.lr
                    self._train_step(batch)
                    self._callback("step_end")

                # validation
                if val_data:
                    self.model.inference()
                    loss, score = self.evaluate_model(*val_data, batch_size=batch_size)
                    self._cache["val_loss"] = loss
                    if self.metric is not None:
                        self._cache[f"val_{self._metric_name}_score"] = score

                self._callback("epoch_end")
                if self._cache["abort"]:
                    break

            self._callback("end")
        finally:
            self.model.clean()

    def evaluate_model(
        self, x: Tensor, y: Tensor, batch_size: int = 32
    ) -> tuple[float, Optional[float]]:
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
        float
            Loss value.
        float, optional
            Metric score.
        """
        dataloader = Dataloader((x, y), batch_size, self.model.device, False)
        losses, scores = [], []

        # compute loss/score for each batch to save memory
        with no_caching():
            for x_batch, y_batch in dataloader():
                y_pred = self.model(x_batch)
                losses.append(self.loss(y_pred, y_batch).item())
                if self.metric is not None:
                    scores.append(self.metric(y_pred, y_batch).item())

        loss = sum(losses) / len(losses)
        if self.metric is not None:
            metric = sum(scores) / len(scores)
            return loss, metric
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
            }[on](self._cache)

    def _train_step(self, batch: tuple[Tensor, ...]) -> None:
        # prepare data
        x, y = batch

        # forward pass
        y_pred = self.model(x)

        # compute loss and metrics
        self._cache["loss"] = self.loss(y_pred, y).item()
        if self.metric is not None:
            key = f"{self._metric_name}_score"
            self._cache[key] = self.metric(y_pred, y).item()

        # backward pass
        self.optimizer.reset_grads()
        self.model.backward(self.loss.backward())

        # update parameters
        self.optimizer.step()
