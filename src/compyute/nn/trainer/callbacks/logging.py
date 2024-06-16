"""Logging callbacks module"""

from typing import Any, Iterable, Literal

from tqdm.auto import tqdm

from .callback import Callback

__all__ = ["History", "ProgressBar"]


class History(Callback):
    """Training history."""

    def __init__(self) -> None:
        self.cache: dict[str, list[float]] = {}

    def __getitem__(self, key) -> list[float]:
        return self.cache[key]

    def _log_stats(self, stats: Iterable[str], trainer_cache: dict[str, Any]) -> None:
        for stat in stats:
            if stat not in trainer_cache:
                return
            if stat in self.cache:
                self.cache[stat].append(trainer_cache[stat])
            else:
                self.cache[stat] = [trainer_cache[stat]]

    def on_step(self, trainer_cache: dict[str, Any]) -> None:
        # get stats that contain scores but are not val scores
        scores = [s for s in trainer_cache.keys() if "score" in s and "val_" not in s]
        self._log_stats(["loss"] + scores, trainer_cache)

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        # get all val stats
        stats = [s for s in trainer_cache.keys() if "val_" in s]
        self._log_stats(stats, trainer_cache)


class ProgressBar(Callback):
    """Callback used for displaying the training progress."""

    def __init__(self, mode: Literal["step", "epoch"] = "step") -> None:
        """Callback used for displaying the training progress.

        Parameters
        ----------
        mode : Literal["step", "epoch"], optional
            Progress bar update mode, by default "step"
            epoch ... one progress bar is shown and updated for each epoch.
            step ... a progress bar is shown per epoch and updated for each step.
        """
        self.pbar = None
        self.mode = mode

    def on_init(self, trainer_cache: dict[str, Any]) -> None:
        if self.mode == "epoch":
            self.pbar = tqdm(unit="epoch", total=trainer_cache["epochs"])

    def on_step(self, trainer_cache: dict[str, Any]) -> None:
        if self.mode == "step":
            self.pbar.update()

    def on_epoch_start(self, trainer_cache: dict[str, Any]) -> None:
        if self.mode == "step":
            self.pbar = tqdm(
                desc=f"Epoch {trainer_cache['t']}/{trainer_cache['epochs']}",
                unit=" steps",
                total=trainer_cache["train_steps"] + 1,
            )

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self._set_pbar_postfix(trainer_cache)
        if self.mode == "epoch":
            self.pbar.update()
        else:
            self.pbar.close()

    def _set_pbar_postfix(self, trainer_cache: dict[str, Any]) -> None:
        stats = []
        for stat in trainer_cache.keys():
            if "loss" not in stat and "score" not in stat:
                continue
            stats.append(f"{stat}={trainer_cache[stat]:.4f}")
        self.pbar.set_postfix_str(", ".join(stats))
