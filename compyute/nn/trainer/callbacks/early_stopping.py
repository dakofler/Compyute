"""Early stopping callback module"""

from .callback import Callback


__all__ = ["EarlyStopping"]


class EarlyStopping(Callback):
    """Early stopping."""

    def __init__(self, patience: int = 3, use_best_params: bool = True) -> None:
        self.patience = patience
        self.use_best_params = use_best_params
        self.state: dict[str, float | int] = {
            "best_loss": float("inf"),
            "best_epoch": 1,
        }

    def epoch(self, trainer) -> None:

        # record best loss
        current_loss = trainer.state["epoch_val_losses"][-1]
        best_loss = self.state["best_loss"]

        if current_loss < best_loss:
            self.state["best_loss"] = current_loss
            self.state["best_epoch"] = trainer.t

        if trainer.t <= self.patience:
            return

        # check if loss has decreased
        if all(
            [
                trainer.state["epoch_val_losses"][-i] > best_loss
                for i in range(1, self.patience + 1)
            ]
        ):
            print(
                f"Early stopping: no improvement over the last {self.patience} epochs (best epoch: {self.state['best_epoch']})"
            )
            trainer.abort = True
