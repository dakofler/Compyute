"""Logging callbacks module"""

from typing import Any, Literal
from tqdm.auto import tqdm
from .callback import Callback


__all__ = ["History", "ProgressBar"]


class History(Callback):
    """Training history."""

    def __init__(self) -> None:
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

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        for stat in state.keys():
            if not stat.startswith("stat_") or "step_" in stat:
                continue
            s = stat.replace("stat_", "")

            if s in self.state:
                self.state[s].append(state[stat])
            else:
                self.state[s] = [state[stat]]


class ProgressBar(Callback):
    """Progress bar."""

    def __init__(self, verbose: Literal[0, 1, 2] = 2) -> None:
        self.pbar = None
        self.verbose = verbose

    def on_init(self, state: dict[str, Any]) -> None:
        if self.verbose != 1:
            return
        self.pbar = tqdm(unit=" epoch", total=state["epochs"])

    def on_step(self, state: dict[str, Any]) -> None:
        if self.verbose != 2:
            return
        self.pbar.update()

    def on_epoch_start(self, state: dict[str, Any]) -> None:
        if self.verbose != 2:
            return
        self.pbar = tqdm(
            desc=f"Epoch {state['t']}/{state["epochs"]}",
            unit=" steps",
            total=state["steps"],
        )

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        match self.verbose:
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
