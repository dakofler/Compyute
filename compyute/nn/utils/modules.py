"""Module utilities."""

from ...base_tensor import _ShapeLike
from ...dtypes import Dtype, _DtypeLike
from ...tensor_ops.creating import ones
from ..modules.module import Module

__all__ = ["get_module_summary"]


def get_module_summary(
    module: Module, input_shape: _ShapeLike, input_dtype: _DtypeLike = Dtype.FLOAT32
) -> str:
    """Returns information about the module and its child modules.

    Parameters
    ----------
    module : Module
        Module to generate a summary of.
    input_shape : _ShapeLike
        Shape of the expected input excluding the batch dimension.
    input_dtype : _DtypeLike, optional
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
                "out_shape": (-1,) + module.y.shape[1:] if module.y is not None else (),
                "n_params": {p.ptr: p.size for p in module.get_parameters(False)},
                "trainable": module.is_trainable,
            }
        )

        # get summary of child modules
        for i, child_module in enumerate(module.modules):
            child_prefix = prefix[:-2]
            if prefix[-2:] == "├─":
                child_prefix += "│ "
            elif prefix[-2:] == "└─":
                child_prefix += "  "
            child_prefix += "└─" if i == len(module.modules) - 1 else "├─"
            build_summary(child_module, child_prefix)

    # perform forward pass to get output shapes
    x = ones((1,) + input_shape, dtype=input_dtype, device=module.device)
    with module.retain_values():
        _ = module(x)

    # get model summary
    module_summaries = []
    build_summary(module, "")
    module.clean()

    # format summary
    divider = "=" * 80
    summary = [
        module.label,
        divider,
        f"{'Layer':30s} {'Output Shape':20s} {'# Parameters':>15s} {'trainable':>12s}",
        divider,
    ]

    n_params = n_train_params = 0
    param_ptrs = []

    for m in module_summaries:
        m_name = m["name"]
        m_out_shape = str(m["out_shape"])
        m_n_params = sum(m["n_params"].values())
        m_trainable = str(m["trainable"])
        summary.append(
            f"{m_name:30s} {m_out_shape:20s} {m_n_params:15d} {m_trainable:>12s}"
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
