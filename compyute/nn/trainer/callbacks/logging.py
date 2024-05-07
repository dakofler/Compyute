"""Logging callbacks module"""

from typing import Any, Literal
from tqdm.auto import tqdm
from .callback import Callback


__all__ = ["History", "ProgressBar"]


class History(Callback):
    """Training history."""

    def __init__(self) -> None:
        self.state: dict[str, list[float]] = {}

    def _log_stat(self, stat: str, state: dict[str, Any]) -> None:
        if stat not in state:
            return
        value = state[stat]
        if stat in self.state:
            self.state[stat].append(value)
        else:
            self.state[stat] = [value]

    def on_step(self, state: dict[str, Any]) -> None:
        self._log_stat("loss", state)
        self._log_stat("score", state)

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        self._log_stat("val_loss", state)
        self._log_stat("val_score", state)


class ProgressBar(Callback):
    """Progress bar."""

    def __init__(self, mode: Literal[0, 1, 2] = 2) -> None:
        self.pbar = None
        self.mode = mode

    def on_init(self, state: dict[str, Any]) -> None:
        if self.mode != 1:
            return
        self.pbar = tqdm(unit=" epoch", total=state["epochs"])

    def on_step(self, state: dict[str, Any]) -> None:
        if self.mode != 2:
            return
        self.pbar.update()

    def on_epoch_start(self, state: dict[str, Any]) -> None:
        if self.mode != 2:
            return
        self.pbar = tqdm(
            desc=f"Epoch {state['t']}/{state["epochs"]}",
            unit=" steps",
            total=state["steps"],
        )

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        match self.mode:
            case 0:
                return
            case 1:
                self._set_pbar_postfix(state)
                self.pbar.update()
            case 2:
                self._set_pbar_postfix(state)
                self.pbar.close()


    def _set_pbar_postfix(self, state: dict[str, Any]) -> None:
        stats = []
        for stat in state.keys():
            if not stat.startswith("stat_"):
                continue
            s = stat.replace("stat_", "")
            stats.append(f"{s}={state[stat]:.4f}")
        self.pbar.set_postfix_str(", ".join(stats))
