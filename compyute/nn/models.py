"""Neural network models module"""

from typing import Callable
import pickle

from tqdm.auto import tqdm
from compyute.functional import concatenate
from compyute.nn.containers import Sequential
from compyute.nn.dataloaders import DataLoader
from compyute.nn.losses import Loss
from compyute.nn.module import Module
from compyute.nn.optimizers.lr_decay import LRDecay
from compyute.nn.optimizers.optimizers import Optimizer
from compyute.tensor import Tensor


__all__ = ["Model", "SequentialModel", "save_model", "load_model"]


class ModelCompilationError(Exception):
    """Error if the model has not been compiled yet."""


class Model(Module):
    """Trainable neural network model."""

    def __init__(self, core_module: Module | None = None) -> None:
        """Trainable neural network model.

        Parameters
        ----------
        core_module : Module, optional
            Core module of the model. For multiple modules use a container as core module.
        """
        super().__init__()
        self.core_module = core_module

        self.optimizer: Optimizer | None = None
        self.lr_decay: LRDecay | None = None
        self.loss_fn: Loss | None = None
        self.metric_fn: Callable[[Tensor, Tensor], Tensor] | None = None
        self._compiled: bool = False

    @property
    def compiled(self) -> bool:
        """Whether the model has been compiled yet."""
        return self._compiled

    def forward(self, x: Tensor) -> Tensor:
        if self.core_module is None:
            raise ValueError(
                "Default forward function cannot be used if no core module is used. If you used Model as a base class, define a custom forward function."
            )

        y = self.core_module.forward(x)

        if self.training:
            self.backward = lambda dy: self.core_module.backward(dy)

        return y

    def compile(
        self,
        optimizer: Optimizer,
        loss_fn: Loss,
        metric_fn: Callable[[Tensor, Tensor], Tensor],
        lr_decay: LRDecay | None = None,
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
        lr_decay : LRDecay | None, optional
            Learning rate decay method to update the optimizers learning rate during training, by default None.
        """

        # for custom models, try to add defined modules to child_model list
        if len(self.child_modules) == 0:
            self.child_modules = [
                i[1] for i in self.__dict__.items() if isinstance(i[1], Module)
            ]

        self._compiled = True
        self.optimizer = optimizer
        optimizer.parameters = self.parameters()

        if lr_decay:
            self.lr_decay = lr_decay
            lr_decay.optimizer = optimizer

        self.loss_fn = loss_fn
        self.metric_fn = metric_fn

    def train(
        self,
        x: Tensor,
        y: Tensor,
        epochs: int = 100,
        verbose: int = 2,
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
        verbose : int, optional
            Mode of reporting intermediate results during training, by default 2.
            0: no reporting
            1: model reports epoch statistics
            2: model reports step statistics
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

        if verbose not in [0, 1, 2]:
            raise AttributeError(f"Invalid verbose mode {verbose}. Can be 0, 1, 2.")

        # create dataloaders
        train_dataloader = DataLoader(x, y, batch_size)
        val_dataloader = DataLoader(*val_data, batch_size) if val_data else None

        train_losses, train_scores = [], []
        val_losses, val_scores = [], []

        if verbose == 1:
            pbar = tqdm(
                unit=" epochs",
                total=epochs,
            )

        for epoch in range(1, epochs + 1):
            # training
            self.training = True
            n_train_steps = len(train_dataloader)

            if verbose == 1:
                pbar.update()
            elif verbose == 2:
                pbar = tqdm(
                    desc=f"Epoch {epoch}/{epochs}",
                    unit=" steps",
                    total=n_train_steps,
                )

            for train_batch in train_dataloader(drop_remaining=True):
                if verbose == 2:
                    pbar.update()

                # prepare data
                x_train_b, y_train_b = train_batch
                x_train_b.to_device(self.device)
                y_train_b.to_device(self.device)

                # forward pass
                train_loss, train_score = self._get_loss_and_score(
                    self.forward(x_train_b), y_train_b
                )
                train_losses.append(train_loss)
                train_scores.append(train_score)

                # backward pass
                self.optimizer.reset_grads()
                self.backward(self.loss_fn.backward())

                # update model parameters
                self.optimizer.step()

            avg_train_loss = sum(train_losses[-n_train_steps:]) / n_train_steps
            avg_train_score = sum(train_scores[-n_train_steps:]) / n_train_steps
            self.training = False

            # update learning rate
            if self.lr_decay:
                self.lr_decay.step()

            # validation
            avg_val_loss = avg_val_score = None
            if val_dataloader is not None:
                n_val_steps = len(val_dataloader)
                epoch_val_losses, epoch_val_scores = [], []

                for val_batch in val_dataloader(shuffle=False, drop_remaining=True):
                    x_val_b, y_val_b = val_batch
                    x_val_b.to_device(self.device)
                    y_val_b.to_device(self.device)

                    val_loss, val_score = self._get_loss_and_score(
                        self.forward(x_val_b), y_val_b
                    )
                    epoch_val_losses.append(val_loss)
                    epoch_val_scores.append(val_score)
                avg_val_loss = sum(epoch_val_losses) / n_val_steps
                avg_val_score = sum(epoch_val_scores) / n_val_steps
                val_losses.append(avg_val_loss)
                val_scores.append(avg_val_score)

            # logging
            if verbose in [1, 2]:
                m = self.metric_fn.__name__
                log = f"train_loss {avg_train_loss:7.4f}, train_{m} {avg_train_score:5.2f}"
                if val_dataloader is not None:
                    log += (
                        f", val_loss {avg_val_loss:7.4f}, val_{m} {avg_val_score:5.2f}"
                    )
                if self.lr_decay is not None:
                    log += f", lr {self.optimizer.lr:.6f}"

                pbar.set_postfix_str(log)
                if verbose == 2:
                    pbar.close()

        if not self.retain_values:
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
            outputs.append(self.forward(x))
        if not self.retain_values:
            self.reset()

        return concatenate(outputs, axis=0)

    def _get_loss_and_score(self, y_pred, y_true) -> tuple[float, float]:
        loss = self.loss_fn(y_pred, y_true).item()
        score = self.metric_fn(y_pred, y_true).item()
        return loss, score

    def to_device(self, device: str) -> None:
        if not self.compiled:
            raise AttributeError("Model must be compiled first")
        super().to_device(device)


class SequentialModel(Model):
    """Sequential model. Layers are processed sequentially."""

    def __init__(self, layers: list[Module]) -> None:
        """Sequential model. Layers are processed sequentially.

        Parameters
        ----------
        layers : list[Module]
            List of layers for the model.
            These layers are processed sequentially starting at index 0.
        """
        super().__init__(Sequential(layers))


def save_model(model: Model, filepath: str) -> None:
    """Saves a model as a binary file.

    Parameters
    ----------
    model : Model
        Model to be saved.
    filepath : str
        Path to the file.
    """
    if not model.compiled:
        raise ModelCompilationError("Model has not been compiled yet.")

    model.to_device("cpu")
    model.optimizer.reset_grads()
    model.loss_fn.backward = None
    model.reset()

    file = open(filepath, "wb")
    pickle.dump(model, file)
    file.close()


def load_model(filepath: str) -> Model:
    """Load a model from a previously saved binary file.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    Model
        Loaded model.
    """
    file = open(filepath, "rb")
    obj = pickle.load(file)
    file.close()
    return obj
