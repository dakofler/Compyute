"""Neural network models module"""

from typing import Literal
from tqdm.auto import tqdm
from .callbacks import Callback
from .optimizers import Optimizer
from .losses import Loss
from .metrics import Metric
from ..dataloaders import DataLoader
from ..models import Model
from ...tensor import Tensor


__all__ = ["Trainer"]


class Trainer:
    """Neural network model trainer."""

    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        loss_functon: Loss,
        metric_function: Metric,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """Neural network model trainer.

        Parameters
        ----------
        model : Model
            Model to be trained.
        optimizer : Optimizer
            Optimizer algorithm used to update model parameters.
        loss_functon : Loss
            Loss function used to evaluate the model.
        metric_function : Metric
            Metric function used to evaluate the model.
        callbacks : list[Callback] | None
            Callback functions to be executed during training, by default None.
        """
        super().__init__()
        self.model = model
        optimizer.parameters = model.parameters
        self.optimizer = optimizer
        self.loss_function = loss_functon
        self.metric_function = metric_function
        self.callbacks = [] if callbacks is None else callbacks

        self.state: dict[str, Tensor | list[float]] = {
            "train_losses": [],
            "train_scores": [],
        }

    def train(
        self,
        X: Tensor,
        y: Tensor,
        epochs: int = 100,
        verbose: Literal[0, 1, 2] = 2,
        val_data: tuple[Tensor, Tensor] | None = None,
        batch_size: int = 1,
    ) -> None:
        """Trains the model using samples and targets.

        Parameters
        ----------
        X : Tensor
            Input tensor.
        y : Tensor
            Target tensor.
        epochs : int, optional
            Number of training iterations, by default 100.
        batch_size : int, optional
            Number of inputs processed in parallel, by default 1.
        verbose : int, optional
            Mode of reporting intermediate results during training, by default 2.
            0: no reporting
            1: model reports epoch statistics
            2: model reports step statistics
        val_dataloader : DataLoader, optional
            Data loader for vaidation data., by default None.
        """

        train_dataloader = DataLoader(X, y, batch_size)
        if val_data:
            val_dataloader = DataLoader(*val_data, batch_size)
            self.state["val_losses"] = []
            self.state["val_scores"] = []

        if verbose == 1:
            pbar = tqdm(unit=" epoch", total=epochs)

        for epoch in range(1, epochs + 1):

            # training
            self.model.training = True
            n_train_steps = len(train_dataloader)

            if verbose == 1:
                pbar.update()
            elif verbose == 2:
                pbar = tqdm(
                    desc=f"Epoch {epoch}/{epochs}",
                    unit=" steps",
                    total=n_train_steps,
                )

            for batch in train_dataloader(drop_remaining=True):
                if verbose == 2:
                    pbar.update()

                # prepare data
                X_batch, y_batch = batch
                X_batch.to_device(self.model.device)
                y_batch.to_device(self.model.device)

                # forward pass
                y_pred = self.model.forward(X_batch)
                train_loss = self.loss_function(y_pred, y_batch).item()
                self.state["train_losses"].append(train_loss)
                train_score = self.metric_function(y_pred, y_batch).item()
                self.state["train_scores"].append(train_score)

                # backward pass
                self.model.backward(self.loss_function.backward())

                # update model parameters
                self.optimizer.step()

                # step callbacks
                for callback in self.callbacks:
                    callback(self, is_step=True)

            self.model.training = False

            # validation
            if val_data:
                retain_values = self.model.retain_values
                self.model.retain_values = False

                for batch in val_dataloader(shuffle=False, drop_remaining=True):
                    # prepare data
                    X_batch, y_batch = batch
                    X_batch.to_device(self.model.device)
                    y_batch.to_device(self.model.device)

                    # forward pass
                    y_pred = self.model.forward(X_batch)
                    val_loss = self.loss_function(y_pred, y_batch).item()
                    self.state["val_losses"].append(val_loss)
                    val_score = self.metric_function(y_pred, y_batch).item()
                    self.state["val_scores"].append(val_score)

                self.model.retain_values = retain_values

            # epoch callbacks
            for callback in self.callbacks:
                callback(self, is_step=False)

            # logging
            if verbose in [1, 2]:
                m = self.metric_function.__class__.__name__
                log = f"train_loss {train_loss:7.4f}, train_{m} {train_score:5.2f}"
                if val_data:
                    log += f", val_loss {val_loss:7.4f}, val_{m} {val_score:5.2f}"

                pbar.set_postfix_str(log)
                if verbose == 2:
                    pbar.close()

        if not self.model.retain_values:
            self.model.reset()

    def evaluate_model(
        self, X: Tensor, y: Tensor, batch_size: int = 1
    ) -> tuple[float, float]:
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
        float
            Loss value.
        float
            Metric score.

        Raises
        ----------
        ModelCompilationError
            If the model has not been compiled yet.
        """
        y.to_device(self.model.device)
        y_pred = self.model.predict(X, batch_size)
        loss = self.loss_function(y_pred, y).item()
        score = self.metric_function(y_pred, y).item()
        return loss, score
