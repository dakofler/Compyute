"""Logging functions."""


__all__ = ["log_training_progress"]


def log_training_progress(
    epoch: int,
    epochs: int,
    time_step: float,
    training_loss: float,
    validation_loss: float | None,
) -> None:
    """Prints out information about intermediate model training results."""
    line = f"epoch {epoch:5d}/{epochs:5d} | time/epoch {time_step:8.2f} ms | loss {training_loss:3.6f}"
    if validation_loss is not None:
        line += f" | val_loss {validation_loss:.6f}"
    print(line)
