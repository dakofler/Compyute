"""Module utilities."""

from typing import Any

from ...tensor_ops.creating import ones
from ...tensors import ShapeLike
from ...typing import DType, float32
from ..modules.module import Module

__all__ = ["get_module_summary"]


def get_module_summary(
    module: Module, input_shape: ShapeLike, input_dtype: DType = float32
) -> str:
    """Returns information about the module and its child modules.

    Parameters
    ----------
    module : Module
        Module to generate a summary of.
    input_shape : _ShapeLike
        Shape of the expected input excluding the batch dimension.
    input_dtype : DType, optional
        Data type of the expected input. Defaults to :class:`compyute.float32`.

    Returns
    -------
    str
        Summary of the module and its child modules.
    """

    def build_summary(module: Module, prefix: str) -> None:
        # add summary of current module
        module_summaries.append(
            {
                "name": prefix + module.label,
                "in_shape": module.x.shape[1:] if module.x else (),
                "out_shape": module.y.shape[1:] if module.y else (),
                "n_params": {p.ptr: p.size for p in module.get_parameters(False)},
                "trainable": module.is_trainable,
            }
        )

        # get summary of child modules
        for i, child_module in enumerate(module.get_modules(recursive=False)):
            child_prefix = prefix[:-2]
            if prefix[-2:] == "├─":
                child_prefix += "│ "
            elif prefix[-2:] == "└─":
                child_prefix += "  "
            child_prefix += "└─" if i == module.n_modules - 1 else "├─"
            build_summary(child_module, child_prefix)

    # perform forward pass to get output shapes
    x = ones((1,) + input_shape, dtype=input_dtype, device=module.device)
    with module.retain_values():
        _ = module(x)

    # get model summary
    module_summaries: list[dict[str, Any]] = []
    build_summary(module, "")
    module.clean()

    # format summary
    divider = "=" * 90
    summary = [
        module.label,
        divider,
        f"{'Layer':30s} {'Input Shape':16s} {'Output Shape':16s} {'# Parameters':>12s} {'trainable':>12s}",
        divider,
    ]

    n_params = n_train_params = 0
    param_ptrs: list[int] = []

    for m in module_summaries:
        m_name = m["name"]
        m_in_shape = str(m["in_shape"])
        m_out_shape = str(m["out_shape"])
        m_n_params = sum(m["n_params"].values())
        m_trainable = str(m["trainable"])
        summary.append(
            f"{m_name:30s} {m_in_shape:16s} {m_out_shape:16s} {m_n_params:12d} {m_trainable:>12s}"
        )

        # count parameters without duplicates (can occur with weight sharing of modules)
        for ptr, n in m["n_params"].items():
            if ptr in param_ptrs:
                continue
            param_ptrs.append(ptr)
            n_params += n
            n_train_params += n if m["trainable"] else 0

    summary.append(divider)
    summary.append(f"Parameters: {n_params}")
    summary.append(f"Trainable parameters: {n_train_params}")

    return "\n".join(summary)
