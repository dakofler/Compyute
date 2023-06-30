"""Logging functions."""


def log_training_progress(
    epoch: int,
    epochs: int,
    time_step: float,
    training_loss: float,
    validation_loss: float | None,
) -> None:
    """_summary_

    Parameters
    ----------
    epoch : int
        _description_
    epochs : int
        _description_
    time_step : float
        _description_
    training_loss : float
        _description_
    validation_loss : float
        _description_
    """
    line = f"epoch {epoch:5d}/{epochs:5d} | time/epoch {time_step:.2f} ms | loss {training_loss:.4f}"
    if validation_loss is not None:
        line += f" | val_loss {validation_loss:.4f}"
    if epochs < 10 or epoch % (epochs // 10) == 0:
        print(line)
