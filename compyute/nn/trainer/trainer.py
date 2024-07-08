"""Neural network models module"""

from typing import Literal, Optional

from ...base_tensor import Tensor
from ...dtypes import _ScalarLike
from ..dataloaders import DataLoader
from ..losses import _LossLike, get_loss_function
from ..metrics import _MetricLike, get_metric_function
from ..modules.module import Module
from ..optimizers import _OptimizerLike, get_optimizer
from .callbacks import Callback

__all__ = ["Trainer"]


class Trainer:
    """Neural network model trainer."""

    __slots__ = (
        "model",
        "optimizer",
        "loss",
        "metric",
        "metric_name",
        "callbacks",
        "cache",
    )

    def __init__(
        self,
        model: Module,
        optimizer: _OptimizerLike,
        loss: _LossLike,
        metric: Optional[_MetricLike] = None,
        callbacks: Optional[list[Callback]] = None,
    ) -> None:
        """Neural network model trainer.

        Parameters
        ----------
        model : Module
            Model to be trained.
        optimizer : _OptimizerLike
            Optimizer algorithm used to update model parameters.
        loss : _LossLike
            Loss function used to evaluate the model.
        metric : _MetricLike, optional
            Metric function used to evaluate the model, by default None.
        callbacks : list[Callback], optional
            Callback functions to be executed during training, by default None.
        """
        super().__init__()
        self.model = model
        self.optimizer = get_optimizer(optimizer)
        self.optimizer.parameters = model.parameters
        self.loss = get_loss_function(loss)
        self.metric = None if metric is None else get_metric_function(metric)
        self.metric_name = None if metric is None else self.metric.__class__.__name__.lower()
        self.callbacks = callbacks
        self.cache: dict = {"abort": False, "t": 1}

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
            Number of training iterations, by default 100.
        val_data : tuple[Tensor, Tensor], optional
            Data used for the validaton every epoch, by default None.
        batch_size : int, optional
            Number of inputs processed in parallel, by default 32.
        """
        train_dataloader = DataLoader(x_train, y_train, batch_size, self.model.device)
        self.cache["t"] = 0
        self.cache["epochs"] = epochs
        self.cache["train_steps"] = len(train_dataloader)
        self._callback("init")

        for _ in range(1, epochs + 1):
            self.cache["t"] += 1
            self._callback("epoch_start")

            # training
            with self.model.training():
                for batch in train_dataloader():
                    self._train_step(batch)
                    self._callback("step")

            # validation
            if val_data:
                loss, score = self.evaluate_model(*val_data, batch_size=batch_size)
                self.cache["val_loss"] = loss
                if self.metric is not None:
                    self.cache[f"val_{self.metric_name}_score"] = score

            self._callback("epoch_end")
            if self.cache["abort"]:
                break

        self.model.cleanup()

    def evaluate_model(
        self, x: Tensor, y: Tensor, batch_size: int = 32
    ) -> tuple[_ScalarLike, Optional[_ScalarLike]]:
        """Evaluates the model using a defined metric.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        y : Tensor
            Target tensor.
        batch_size : int, optional
            Number of inputs processed in parallel, by default 32.

        Returns
        ----------
        ScalarLike
            Loss value.
        ScalarLike, optional
            Metric score.
        """
        dataloader = DataLoader(
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

    def _callback(self, on: Literal["init", "step", "epoch_start", "epoch_end"]) -> None:
        if self.callbacks is None:
            return
        for callback in self.callbacks:
            {
                "init": callback.on_init,
                "step": callback.on_step,
                "epoch_start": callback.on_epoch_start,
                "epoch_end": callback.on_epoch_end,
            }[on](self.cache)

    def _train_step(self, batch: tuple[Tensor, Tensor]) -> None:
        # prepare data
        x_batch, y_batch = batch

        # forward pass
        y_pred = self.model(x_batch)

        # compute loss and metrics
        self.cache["loss"] = self.loss(y_pred, y_batch).item()
        if self.metric is not None:
            self.cache[f"{self.metric_name}_score"] = self.metric(y_pred, y_batch).item()

        # backward pass
        self.optimizer.reset_grads()
        self.model.backward(self.loss.backward())

        # update parameters
        self.optimizer.step()
