"""History callback module"""

from typing import Any
from tqdm.auto import tqdm
from .callback import Callback


__all__ = ["History"]


class History(Callback):
    """Training history."""

    def __init__(self) -> None:
        """Keeps a history of losses and metrics.

        Parameters
        ----------
        metric : str, optional
            Metric to log, by default None.
        """
        self.state: dict[str, list[float]] = {}

    def on_step(self, state: dict[str, Any]) -> None:
        for stat in state.keys():
            if not stat.startswith("stat_step_"):
                continue
            s = stat.replace("stat_", "")

            if s in self.state:
                self.state[s].append(state[stat])
            else:
                self.state[s] = [state[stat]]

    def on_epoch(self, state: dict[str, Any]) -> None:
        for stat in state.keys():
            if not stat.startswith("stat_") or "step_" in stat:
                continue
            s = stat.replace("stat_", "")

            if s in self.state:
                self.state[s].append(state[stat])
            else:
                self.state[s] = [state[stat]]


class ProgressBar(Callback):
    """Training history."""

    def on_step(self, state: dict[str, Any]) -> None: ...

    def on_epoch(self, state: dict[str, Any]) -> None: ...

    # if verbose == 1:
    #     pbar = tqdm(unit=" epoch", total=epochs)

    # if verbose == 2:
    #     pbar.update()

    # if verbose == 1:
    #     pbar.update()
    # elif verbose == 2:
    #     pbar = tqdm(
    #         desc=f"Epoch {epoch}/{epochs}",
    #         unit=" steps",
    #         total=n_train,
    #     )

    # if verbose in [1, 2]:
    #     include_val = val_data is not None
    #     pbar.set_postfix_str(self.__get_pbar_postfix(include_val))
    #     if verbose == 2:
    #         pbar.close()

    # def __get_pbar_postfix(self, include_val: bool = False) -> str:
    #     loss = self.state["loss"][-1]
    #     log = f"train_loss {loss:7.4f}"

    #     if self.metric is not None:
    #         score = self.state[f"{self.metric.name}"][-1]
    #         log += f", train_{self.metric.name} {score:5.2f}"

    #     if include_val:
    #         val_loss = self.state["val_loss"][-1]
    #         log += f", val_loss {val_loss:7.4f}"

    #         if self.metric is not None:
    #             val_score = self.state[f"val_{self.metric.name}"][-1]
    #             log += f", val_{self.metric.name} {val_score:5.2f}"

    #     return log
