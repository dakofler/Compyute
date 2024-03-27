"""Neural network models module"""

from typing import Literal
from tqdm.auto import tqdm
from .callbacks import Callback
from .optimizers import Optimizer, get_optim_from_str
from .losses import Loss, get_loss_from_str
from .metrics import Metric, get_metric_from_str
from ..dataloaders import DataLoader
from ..models import Model
from ...tensor import Tensor
from ...types import ScalarLike


__all__ = ["Trainer"]


class Trainer:
    """Neural network model trainer."""

    def __init__(
        self,
        model: Model,
        optimizer: Optimizer | Literal["sgd", "adam", "adamw", "nadam"],
        loss_functon: Loss | Literal["mse", "crossentropy"],
        metric_function: Metric | Literal["accuracy", "r2"] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """Neural network model trainer.

        Parameters
        ----------
        model : Model
            Model to be trained.
        optimizer : Optimizer | Literal["sgd", "adam", "adamw", "nadam"]
            Optimizer algorithm used to update model parameters.
        loss_functon : Loss | Literal["mse", "crossentropy"]
            Loss function used to evaluate the model.
        metric_function : Metric | Literal["accuracy", "r2"] | None, optional
            Metric function used to evaluate the model, by default None.
        callbacks : list[Callback] | None
            Callback functions to be executed during training, by default None.
        """
        super().__init__()
        self.model = model

        self.optimizer = (
            optimizer
            if isinstance(optimizer, Optimizer)
            else get_optim_from_str(optimizer)
        )
        self.optimizer.parameters = model.parameters

        self.loss_function = (
            loss_functon
            if isinstance(loss_functon, Loss)
            else get_loss_from_str(loss_functon)
        )

        self.metric_function = (
            metric_function
            if isinstance(metric_function, Metric | None)
            else get_metric_from_str(metric_function)
        )

        self.callbacks = [] if callbacks is None else callbacks

        self.state: dict[str, Tensor | list[float]] = {"loss": [], "epoch_loss": []}
        if metric_function is not None:
            self.state[self.metric_function.name] = []
            self.state[f"epoch_{self.metric_function.name}"] = []

        self.t: int = 1
        self.abort: bool = False

    def train(
        self,
        X_train: Tensor,
        y_train: Tensor,
        epochs: int = 100,
        verbose: Literal[0, 1, 2] = 2,
        val_data: tuple[Tensor, Tensor] | None = None,
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
        val_data : tuple[Tensor, Tensor] | None, optional
            Data used for the validaton every epoch, by default None.
        batch_size : int, optional
            Number of inputs processed in parallel, by default 32.
        """

        train_dl = DataLoader(X_train, y_train, batch_size)

        if val_data:
            val_dl = DataLoader(*val_data, batch_size)
            self.state["val_loss"] = []
            self.state["epoch_val_loss"] = []

            if self.metric_function is not None:
                self.state[f"val_{self.metric_function.name}"] = []
                self.state[f"epoch_val_{self.metric_function.name}"] = []

        if verbose == 1:
            pbar = tqdm(unit=" epoch", total=epochs)

        for epoch in range(self.t, self.t + epochs):

            # training
            self.model.training = True
            n_train = len(train_dl)

            if verbose == 1:
                pbar.update()
            elif verbose == 2:
                pbar = tqdm(
                    desc=f"Epoch {epoch}/{epochs}",
                    unit=" steps",
                    total=n_train,
                )

            for batch in train_dl(drop_remaining=True):
                if verbose == 2:
                    pbar.update()
                self.train_step(batch)
                for callback in self.callbacks:
                    callback(self, is_step=True)

            # train statistics
            self.__log_epoch_loss(n_train)
            if self.metric_function is not None:
                self.__log_epoch_score(n_train)

            self.model.training = False

            # validation
            if val_data:
                retain_values = self.model.retain_values
                self.model.retain_values = False
                n_val = len(val_dl)

                for batch in val_dl(shuffle=False, drop_remaining=True):
                    self.val_step(batch)

                # val statistics
                self.__log_epoch_loss(n_val, "val_")
                if self.metric_function is not None:
                    self.__log_epoch_score(n_val, "val_")

                self.model.retain_values = retain_values

            # logging #n_train_steps
            if verbose in [1, 2]:
                include_val = val_data is not None
                pbar.set_postfix_str(self.__get_pbar_postfix(include_val))
                if verbose == 2:
                    pbar.close()

            # epoch callbacks
            for callback in self.callbacks:
                callback(self, is_step=False)

            if self.abort:
                break

            self.t += 1

        if not self.model.retain_values:
            self.model.reset()

    def __get_pbar_postfix(self, include_val: bool = False) -> str:
        loss = self.state["epoch_loss"][-1]
        log = f"train_loss {loss:7.4f}"

        if self.metric_function is not None:
            metric = self.metric_function.name
            score = self.state[f"epoch_{metric}"][-1]
            log += f", train_{metric} {score:5.2f}"

        if include_val:
            val_loss = self.state["epoch_val_loss"][-1]
            log += f", val_loss {val_loss:7.4f}"
            if self.metric_function is not None:
                val_score = self.state[f"epoch_val_{metric}"][-1]
                log += f", val_{metric} {val_score:5.2f}"

        return log

    def train_step(self, batch: tuple[Tensor, Tensor]) -> None:
        """Performs one training step on a batch of data.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            Batch of data.
        """
        X_batch, y_batch = batch
        X_batch.to_device(self.model.device)
        y_batch.to_device(self.model.device)

        # forward pass
        y_pred = self.model.forward(X_batch)
        loss = self.__get_loss(y_pred, y_batch)
        self.__log_loss(loss)

        if self.metric_function is not None:
            score = self.__get_score(y_pred, y_batch)
            self.__log_score(score)

        # backward pass
        self.model.backward(self.loss_function.backward())

        # update model parameters
        self.optimizer.step()

    def val_step(self, batch: tuple[Tensor, Tensor]) -> None:
        """Performs one validation step on a batch of data.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            Batch of data.
        """
        X_batch, y_batch = batch
        X_batch.to_device(self.model.device)
        y_batch.to_device(self.model.device)

        # forward pass
        y_pred = self.model.forward(X_batch)
        loss = self.__get_loss(y_pred, y_batch)
        self.__log_loss(loss, prefix="val_")

        if self.metric_function is not None:
            score = self.__get_score(y_pred, y_batch)
            self.__log_score(score, prefix="val_")

    def __get_loss(self, y_pred: Tensor, y_true: Tensor) -> ScalarLike:
        return self.loss_function(y_pred, y_true).item()

    def __log_loss(self, loss: float, prefix: str = "") -> None:
        self.state[f"{prefix}loss"].append(loss)

    def __log_epoch_loss(self, n_steps: int, prefix: str = "") -> None:
        epoch_loss = sum(self.state[f"{prefix}loss"][-n_steps:]) / n_steps
        self.state[f"epoch_{prefix}loss"].append(epoch_loss)

    def __get_score(self, y_pred: Tensor, y_true: Tensor) -> ScalarLike | None:
        return self.metric_function(y_pred, y_true).item()

    def __log_score(self, score: float, prefix: str = "") -> None:
        self.state[f"{prefix}{self.metric_function.name}"].append(score)

    def __log_epoch_score(self, n_steps: int, prefix: str = "") -> None:
        epoch_loss = (
            sum(self.state[f"{prefix}{self.metric_function.name}"][-n_steps:]) / n_steps
        )
        self.state[f"epoch_{prefix}{self.metric_function.name}"].append(epoch_loss)

    def evaluate_model(
        self, X: Tensor, y: Tensor, batch_size: int = 1
    ) -> tuple[ScalarLike, ScalarLike]:
        """Evaluates the model using a defined metric.

        Parameters
        ----------
        X : Tensor
            Input tensor.
        y : Tensor
            Target tensor.
        batch_size : int, optional
            Number of inputs processed in parallel, by default 1.

        Returns
        ----------
        ScalarLike
            Loss value.
        ScalarLike
            Metric score.

        Raises
        ----------
        ModelCompilationError
            If the model has not been compiled yet.
        """
        y.to_device(self.model.device)
        y_pred = self.model.predict(X, batch_size)
        loss = self.__get_loss(y_pred, y)
        score = self.__get_score(y_pred, y)
        return loss, score
