"""Logging functions."""

# from walnut.nn.models import Model
from walnut.tensor import ShapeLike
import walnut.tensor_utils as tu


__all__ = ["log_training_progress", "summary"]


def log_training_progress(
    epoch: int,
    epochs: int,
    time_step: float,
    training_loss: float,
    validation_loss: float | None,
) -> None:
    """Prints out information about intermediate model training results."""
    line = f"epoch {epoch:5d}/{epochs:5d} | step {time_step:8.2f} ms | loss {training_loss:3.6f}"
    if validation_loss is not None:
        line += f" | val_loss {validation_loss:.6f}"
    print(line)


def summary(model, input_shape: ShapeLike, input_dtype: str = "float") -> None:
    """Prints information about a walnut.nn model.

    Parameters
    ----------
    model : Model
        Neural network model.
    input_shape : ShapeLike
        Shape of the model input ignoring the batch dimension.
    input_dtype : str, optional
        Input data type, by default "float".
    """
    n = 57
    string = "-" * n
    string += f"\n{'Layer (type)':20s} {'Output Shape':20s} {'# Parameters':>15s}\n"
    string += "=" * n
    string += "\n"
    tot_parameters = 0

    x = tu.ones((1,) + input_shape).astype(input_dtype)
    _ = model(x)

    for layer in model.layers:
        name = layer.__class__.__name__
        output_shape = str((-1,) + layer.y.shape[1:])
        num_parameters = sum(p.data.size for p in layer.parameters)
        tot_parameters += num_parameters
        string += f"{name:20s} {output_shape:20s} {num_parameters:15d}\n"

    string += "=" * n
    print(f"{string}\n\nTotal parameters: {tot_parameters}")
