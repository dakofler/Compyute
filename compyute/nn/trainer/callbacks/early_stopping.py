"""Early stopping callback module"""

from typing import Any
from .callback import Callback


__all__ = ["EarlyStopping"]


class EarlyStopping(Callback):
    """Early stopping."""

    def __init__(
        self,
        patience: int = 3,
        use_best_params: bool = True,
        target: str = "loss",
    ) -> None:
        """Aborts the training process if the model stops improving.

        Parameters
        ----------
        patience : int, optional
            Number of epocs without improvement, before the training is aborted, by default 3.
        use_best_params : bool, optional
            Whether to reset the model parameters to the best values found, by default True.
        target : str, optional
            Metric to consider, by default "loss".
        """
        self.patience = patience
        self.use_best_params = use_best_params
        self.target = target
        self.state: dict[str, Any] = {"best_epoch": 1, "best_loss": float("inf"), "history": []}

    def on_epoch(self, state: dict[str, Any]) -> None:
        self.state["history"].append(state[f"stat_{self.target}"])
        best_loss = self.state["best_loss"]

        # record best loss
        if self.state["history"][-1] < best_loss:
            self.state["best_epoch"] = state["t"]
            self.state["best_loss"] = self.state["history"][-1]

            # save best parameters
            if self.use_best_params:
                self.state["best_params"] = [p.copy() for p in state["model"].parameters]

        if len(self.state["history"]) <= self.patience:
            return

        # check if loss has decreased
        if all([self.state["history"][-i] > best_loss for i in range(1, self.patience + 1)]):
            msg = f"Early stopping: no improvement over last {self.patience} epochs."

            # reset model parameters to best epoch
            if self.use_best_params:
                best_epoch = self.state["best_epoch"]
                msg += f" Resetting parameters best epoch {best_epoch}."
                for i, p in enumerate(state["model"].parameters):
                    p.data = self.state["best_params"][i].data

            print(msg)
            state["abort"] = True
