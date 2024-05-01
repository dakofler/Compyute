"""Neural network models module"""

from typing import Literal, Optional
from tqdm.auto import tqdm
from .callbacks import Callback
from .optimizers import Optimizer, get_optimizer
from .losses import Loss, get_loss
from .metrics import Metric, get_metric
from ..dataloaders import DataLoader, batched
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
        loss: Loss | Literal["binary_crossentropy", "crossentropy", "mse"],
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
        self.state: dict[str, Tensor | list[float]] = {"loss": [], "epoch_loss": []}

        if self.metric is not None:
            self.state[self.metric.name] = []
            self.state[f"epoch_{self.metric.name}"] = []

        self.abort: bool = False
        self.t: int = 0

    def train(
        self,
        X_train: Tensor,
        y_train: Tensor,
        epochs: int = 100,
        verbose: Literal[0, 1, 2] = 2,
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
        verbose : int, optional
            Mode of reporting intermediate results during training, by default 2.
            0: no reporting
            1: model reports epoch statistics
            2: model reports step statistics
        val_data : tuple[Tensor, Tensor], optional
            Data used for the validaton every epoch, by default None.
        batch_size : int, optional
            Number of inputs processed in parallel, by default 32.
        """
        self.t = 0
        train_dataloader = DataLoader(X_train, y_train, batch_size, self.model.device)

        if val_data:
            val_dataloader = DataLoader(*val_data, batch_size, self.model.device, False, False)
            self.state["val_loss"] = []
            self.state["epoch_val_loss"] = []

            if self.metric is not None:
                self.state[f"val_{self.metric.name}"] = []
                self.state[f"epoch_val_{self.metric.name}"] = []

        if verbose == 1:
            pbar = tqdm(unit=" epoch", total=epochs)

        for epoch in range(1, epochs + 1):
            self.t += 1

            # training
            self.model.set_training(True)
            n_train = len(train_dataloader)

            if verbose == 1:
                pbar.update()
            elif verbose == 2:
                pbar = tqdm(
                    desc=f"Epoch {epoch}/{epochs}",
                    unit=" steps",
                    total=n_train,
                )

            for batch in train_dataloader():
                if verbose == 2:
                    pbar.update()

                self.__train_step(batch)

                for callback in self.callbacks:
                    callback.on_step(self)

            # statistics
            self.__log_epoch_loss(n_train)
            self.__log_epoch_score(n_train)

            self.model.set_training(False)

            # validation
            if val_data:
                retain_values = self.model.retain_values
                self.model.set_retain_values(False)

                for batch in val_dataloader():
                    self.__val_step(batch)

                # statistics
                n_val = len(val_dataloader)
                self.__log_epoch_loss(n_val, "val_")
                self.__log_epoch_score(n_val, "val_")

                self.model.set_retain_values(retain_values)

            # logging #n_train_steps
            if verbose in [1, 2]:
                include_val = val_data is not None
                pbar.set_postfix_str(self.__get_pbar_postfix(include_val))
                if verbose == 2:
                    pbar.close()

            # epoch callbacks
            for callback in self.callbacks:
                callback.on_epoch(self)

            if self.abort:
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

        # make predictions
        y_pred = batched(self.model.forward, batch_size, self.model.device, False)(x)
        y.to_device(self.model.device)
        loss = self.__compute_loss(y_pred, y)
        score = self.__compute_score(y_pred, y)
        return loss, score

    def __train_step(self, batch: tuple[Tensor, Tensor]) -> None:
        """Performs one training step on a batch of data.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            Batch of data.
        """

        # prepare data
        X_batch, y_batch = batch
        X_batch.to_device(self.model.device)
        y_batch.to_device(self.model.device)

        # forward pass
        y_pred = self.model.forward(X_batch)
        _ = self.__compute_loss(y_pred, y_batch, log=True)
        _ = self.__compute_score(y_pred, y_batch, log=True)

        # backward pass
        self.model.backward(self.loss.backward())

        # update parameters
        self.optimizer.step()

    def __val_step(self, batch: tuple[Tensor, Tensor]) -> None:
        """Performs one validation step on a batch of data.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            Batch of data.
        """

        # prepare data
        X_batch, y_batch = batch
        X_batch.to_device(self.model.device)
        y_batch.to_device(self.model.device)

        # forward pass
        y_pred = self.model.forward(X_batch)
        _ = self.__compute_loss(y_pred, y_batch, log=True, log_prefix="val_")
        _ = self.__compute_score(y_pred, y_batch, log=True, log_prefix="val_")

    def __compute_loss(
        self, y_pred: Tensor, y_true: Tensor, log: bool = False, log_prefix: str = ""
    ) -> ScalarLike:
        loss = self.loss(y_pred, y_true).item()
        if log:
            self.state[f"{log_prefix}loss"].append(loss)
        return loss

    def __compute_score(
        self, y_pred: Tensor, y_true: Tensor, log: bool = False, log_prefix: str = ""
    ) -> Optional[ScalarLike]:
        if self.metric is None:
            return
        score = self.metric(y_pred, y_true).item()
        if log:
            self.state[f"{log_prefix}{self.metric.name}"].append(score)
        return score

    def __log_epoch_loss(self, n_steps: int, prefix: str = "") -> None:
        avg_loss = sum(self.state[f"{prefix}loss"][-n_steps:]) / n_steps
        self.state[f"epoch_{prefix}loss"].append(avg_loss)

    def __log_epoch_score(self, n_steps: int, prefix: str = "") -> None:
        if self.metric is None:
            return
        avg_score = sum(self.state[f"{prefix}{self.metric.name}"][-n_steps:]) / n_steps
        self.state[f"epoch_{prefix}{self.metric.name}"].append(avg_score)

    def __get_pbar_postfix(self, include_val: bool = False) -> str:
        loss = self.state["epoch_loss"][-1]
        log = f"train_loss {loss:7.4f}"

        if self.metric is not None:
            score = self.state[f"epoch_{self.metric.name}"][-1]
            log += f", train_{self.metric.name} {score:5.2f}"

        if include_val:
            val_loss = self.state["epoch_val_loss"][-1]
            log += f", val_loss {val_loss:7.4f}"

            if self.metric is not None:
                val_score = self.state[f"epoch_val_{self.metric.name}"][-1]
                log += f", val_{self.metric.name} {val_score:5.2f}"

        return log
