"""Neural network models module"""

from typing import Any, Literal, Optional
from .callbacks import Callback
from .optimizers import Optimizer, get_optimizer
from .losses import Loss, get_loss
from .metrics import Metric, get_metric
from ..dataloaders import DataLoader
from ..modules import Module
from ...tensor import Tensor
from ...types import ScalarLike


__all__ = ["Trainer"]


class Trainer:
    """Neural network model trainer."""

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer | Literal["sgd", "adam", "adamw", "nadam"],
        loss: Loss | Literal["binary_cross_entropy", "cross_entropy", "mean_squared_error"],
        metric: Optional[Metric | Literal["accuracy", "r2"]] = None,
        callbacks: Optional[list[Callback]] = None,
    ) -> None:
        """Neural network model trainer.

        Parameters
        ----------
        model : Module
            Model to be trained.
        optimizer : Optimizer | Literal["sgd", "adam", "adamw", "nadam"]
            Optimizer algorithm used to update model parameters.
        loss : Loss | Literal["mse", "crossentropy"]
            Loss function used to evaluate the model.
        metric : Metric | Literal["accuracy", "r2"], optional
            Metric function used to evaluate the model, by default None.
        callbacks : list[Callback], optional
            Callback functions to be executed during training, by default None.
        """
        super().__init__()
        self.model = model
        self.optimizer = get_optimizer(optimizer)
        self.optimizer.parameters = model.parameters
        self.loss = get_loss(loss)
        self.metric = None if metric is None else get_metric(metric)
        self.callbacks = [] if callbacks is None else callbacks

        self._callback_cache: dict[str, Any] = {
            "abort": False,
            "model": self.model,
            "optimizer": self.optimizer,
            "t": 0,
        }

    def train(
        self,
        X_train: Tensor,
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
        train_dataloader = DataLoader(X_train, y_train, batch_size, self.model.device)
        self._callback_cache["t"] = 0
        self._callback_cache["epochs"] = epochs
        self._callback_cache["steps"] = len(train_dataloader)
        self._callback("init")

        for _ in range(1, epochs + 1):
            self._callback_cache["t"] += 1
            self._callback("epoch_start")

            # training
            self.model.set_training(True)

            for batch in train_dataloader():
                self._train_step(batch)
                self._callback("step")

            self.model.set_training(False)

            # validation
            if val_data:
                loss, score = self.evaluate_model(*val_data, batch_size=batch_size)
                self._callback_cache["val_loss"] = loss
                self._callback_cache["val_score"] = score

            self._callback("epoch_end")
            if self._callback_cache["abort"]:
                break

        if not self.model.retain_values:
            self.model.reset()

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

        losses = []
        if self.metric is not None:
            scores = []

        for x_batch, y_batch in dataloader():
            y_pred = self.model.forward(x_batch)
            losses.append(self.loss(y_pred, y_batch).item())
            if self.metric is not None:
                scores.append(self.metric(y_pred, y_batch).item())

        loss = sum(losses) / len(losses)
        if self.metric is not None:
            return loss, sum(scores) / len(scores)
        return loss, None

    def _callback(self, on: Literal["init", "step", "epoch_start", "epoch_end"]) -> None:
        for callback in self.callbacks:
            match on:
                case "init":
                    callback.on_init(self._callback_cache)
                case "step":
                    callback.on_step(self._callback_cache)
                case "epoch_start":
                    callback.on_epoch_start(self._callback_cache)
                case "epoch_end":
                    callback.on_epoch_end(self._callback_cache)

    def _train_step(self, batch: tuple[Tensor, Tensor]) -> None:
        # prepare data
        X_batch, y_batch = batch

        # forward pass
        y_pred = self.model.forward(X_batch)

        # compute loss
        self._callback_cache["loss"] = self.loss(y_pred, y_batch).item()
        if self.metric is not None:
            self._callback_cache["score"] = self.metric(y_pred, y_batch).item()

        # backward pass
        self.model.backward(self.loss.backward())

        # update parameters
        self.optimizer.step()
