"""Neural network models module"""

import time

from walnut import tensor_utils as tu
from walnut.tensor import Tensor, NumpyArray
from walnut.logger import log_training_progress
from walnut.nn.losses import Loss
from walnut.nn.metrics import Metric
from walnut.nn.module import Module, ModuleCompilationError
from walnut.nn.optimizers import Optimizer


__all__ = ["Sequential"]


class Model(Module):
    """Neural network model base class."""

    def __init__(self, layers: list[Module]) -> None:
        super().__init__()
        self.layers = layers
        self.optimizer: Optimizer | None = None
        self.loss_fn: Loss | None = None
        self.metric: Metric | None = None

    def __repr__(self) -> str:
        string = self.__class__.__name__ + "\n\n"
        for layer in self.layers:
            string += layer.__repr__() + "\n"
        sum_params = sum(p.data.size for p in self.get_parameters())
        string += f"\ntotal trainable parameters {sum_params}"
        return string

    def compile(self, optimizer: Optimizer, loss_fn: Loss, metric: Metric) -> None:
        """Compiles the model.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer algorithm to be used to update parameters.
        loss_fn : Loss
            Loss function to be used to compute losses and gradients.
        metric : Metric
            Metric to be used to evaluate the model.
        """
        self.optimizer = optimizer
        optimizer.parameters = self.get_parameters()
        self.loss_fn = loss_fn
        self.metric = metric

    def train(
        self,
        x: Tensor,
        y: Tensor,
        epochs: int = 100,
        batch_size: int | None = None,
        verbose: str = "reduced",
        val_data: tuple[Tensor, Tensor] | None = None,
        reset_params: bool = True,
    ) -> tuple[list[float], list[float]]:
        """Trains the model using samples and targets.

        Parameters
        ----------
        x : Tensor
            Tensor of input values (features).
        y : Tensor
            Tensor of target values.
        epochs : int, optional
            Number of training iterations, by default 100.
        batch_size : int | None, optional
            Number of training samples used per epoch, by default None.
            If None, all samples are used.
        verbose : str, optional
            Whether to print out intermediate results while training, by default "reduced".
        val_data : tuple[Tensor, Tensor] | None, optional
            Data used for validation during training, by default None.
        reset_params : bool, optional
            Whether to reset grads after training. Improves memory usage, by default True.

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
        if not self.loss_fn:
            raise ModuleCompilationError("Model is not compiled yet.")

        train_loss_history = []
        val_loss_history = []

        for epoch in range(0, epochs + 1):
            start = time.time()
            x_train, y_train = tu.shuffle(x, y, batch_size)
            self.set_training(True)

            # forward pass
            predictions = self(x_train)

            # compute loss
            train_loss = self.loss_fn(predictions, y_train).item()
            train_loss_history.append(train_loss)

            # backward pass
            y_grad = self.loss_fn.backward()
            self.backward(y_grad)

            self.optimizer.step()
            self.set_training(False)

            # validation
            val_loss = None
            if val_data is not None:
                x_val, y_val = val_data
                val_predictions = self(x_val)
                val_loss = self.loss_fn(val_predictions, y_val).item()
                val_loss_history.append(val_loss)

            end = time.time()
            step = round((end - start) * 1000.0, 2)
            log_training_progress(verbose, epoch, epochs, step, train_loss, val_loss)

        # reset parameters to improve memory efficiency
        if reset_params:
            self.reset_grads()

        return train_loss_history, val_loss_history

    def evaluate(self, x: Tensor, y: Tensor) -> tuple[float, float]:
        """Evaluates the model using a defined metric.

        Parameters
        ----------
        x : Tensor
            Tensor of input values (features).
        y : Tensor
            Tensor of target values.

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
        if self.metric is None or self.loss_fn is None:
            raise ModuleCompilationError("Model not compiled yet.")

        predictions = self(x)
        loss = self.loss_fn(predictions, y).item()
        score = self.metric(predictions, y)
        return loss, score

    def set_training(self, on: bool = False) -> None:
        """Sets the mode for all model layers."""
        for layer in self.layers:
            layer.training = on

    def get_parameters(self) -> list[Tensor]:
        """Returns trainable parameters of a models layers."""
        parameters = []
        for layer in self.layers:
            parameters += layer.parameters
        return parameters

    def reset_grads(self) -> None:
        """Resets gradients to reduce memory usage."""
        for layer in self.layers:
            layer.reset_grads()


class Sequential(Model):
    """Feed forward neural network model."""

    def __init__(self, layers: list[Module]) -> None:
        super().__init__(layers)

        def backward(y_grad: NumpyArray) -> NumpyArray:
            layers_reversed = self.layers.copy()
            layers_reversed.reverse()

            for layer in layers_reversed:
                y_grad = layer.backward(y_grad)
            return y_grad

        self.backward = backward

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
