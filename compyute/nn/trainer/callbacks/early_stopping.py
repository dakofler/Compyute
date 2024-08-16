"""Early stopping callbacks."""

from typing import Any

from ...modules import Module
from .callback import Callback

__all__ = ["EarlyStopping"]


class EarlyStopping(Callback):
    """Aborts the training process if the model stops improving.

    Parameters
    ----------
    model : Module
        Model to be trained.
    patience : int, optional
        Number of epocs without improvement, before the training is aborted. Defaults to ``3``.
    use_best_params : bool, optional
        Whether to reset the model parameters to the best values found. Defaults to ``True``.
    target : str, optional
        Metric to consider. Defaults to ``loss``.
    """

    def __init__(
        self,
        model: Module,
        patience: int = 3,
        use_best_params: bool = True,
        target: str = "loss",
    ) -> None:
        self.model = model
        self.patience = patience
        self.use_best_params = use_best_params
        self.target = target
        self.cache: dict[str, Any] = {
            "best_epoch": 1,
            "best_loss": float("inf"),
            "history": [],
        }

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self.cache["history"].append(trainer_cache[self.target])
        best_loss = self.cache["best_loss"]

        # record best loss
        if self.cache["history"][-1] < best_loss:
            self.cache["best_epoch"] = trainer_cache["t"]
            self.cache["best_loss"] = self.cache["history"][-1]

            # save best parameters
            if self.use_best_params:
                self.cache["best_params"] = [
                    p.copy() for p in self.model.get_parameters()
                ]

        if len(self.cache["history"]) <= self.patience:
            return

        # check if loss has decreased
        if all(
            [self.cache["history"][-i] > best_loss for i in range(1, self.patience + 1)]
        ):
            msg = f"Early stopping: no improvement over last {self.patience} epochs."

            # reset model parameters to best epoch
            if self.use_best_params:
                best_epoch = self.cache["best_epoch"]
                msg += f" Resetting parameters best epoch {best_epoch}."
                for i, p in enumerate(self.model.get_parameters()):
                    p.data = self.cache["best_params"][i].data

            print(msg)
            trainer_cache["abort"] = True
