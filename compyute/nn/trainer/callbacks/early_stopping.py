"""Early stopping callback module"""

from .callback import Callback
from ....tensor import Tensor


__all__ = ["EarlyStopping"]


class EarlyStopping(Callback):
    """Early stopping."""

    def __init__(self, patience: int = 3, use_best_params: bool = True) -> None:
        """Aborts the training process if the model stops improving.

        Parameters
        ----------
        patience : int, optional
            Number of epocs without improvement, before the training is aborted, by default 3.
        use_best_params : bool, optional
            Whether to reset the model parameters to the best values found, by default True.
        """
        self.patience = patience
        self.use_best_params = use_best_params
        self.state: dict[str, float | int | list[Tensor]] = {"best_loss": float("inf")}

    def epoch(self, trainer) -> None:
        loss_hist = trainer.state["epoch_val_losses"]
        best_loss = self.state["best_loss"]

        # record best loss
        if loss_hist[-1] < best_loss:
            self.state["best_loss"] = loss_hist[-1]
            self.state["best_epoch"] = trainer.t

            # save best parameters
            if self.use_best_params:
                self.state["best_params"] = [p.copy() for p in trainer.model.parameters]

        if trainer.t <= self.patience:
            return

        # check if loss has decreased
        if all([loss_hist[-i] > best_loss for i in range(1, self.patience + 1)]):
            msg = f"Early stopping: no improvement over last {self.patience} epochs."

            if self.use_best_params:
                msg += f"\nResetting parameters to epoch {self.state['best_epoch']}."

                # reset model parameters
                for i, p in enumerate(trainer.model.parameters):
                    p.data = self.state["best_params"][i].data

            print(msg)
            trainer.abort = True
