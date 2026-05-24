from collections import defaultdict
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module


def _foreach_norm(grads: list[Tensor], norm_type: float) -> list[Tensor]:
    if hasattr(torch, "_foreach_norm"):
        try:
            return torch._foreach_norm(grads, norm_type)
        except RuntimeError:
            pass
    return [grad.norm(norm_type) for grad in grads]


def _validate_norm_type(norm_type: Union[float, int, str]) -> float:
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(
            f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}"
        )
    return norm_type


def _total_norm_from_group_norms(
    norm_groups: list[list[Tensor]], norm_type: float
) -> Tensor | None:
    group_totals = []
    for group_norms in norm_groups:
        if not group_norms:
            continue
        group_device = group_norms[0].device
        stacked = torch.stack([norm.to(group_device) for norm in group_norms])
        if norm_type == float("inf"):
            group_totals.append(stacked.max())
        else:
            group_totals.append(stacked.norm(norm_type))

    if not group_totals:
        return None

    total_device = group_totals[0].device
    stacked_group_totals = torch.stack(
        [group_total.to(total_device) for group_total in group_totals]
    )
    if norm_type == float("inf"):
        total_norm = stacked_group_totals.max()
    else:
        total_norm = stacked_group_totals.norm(norm_type)
    return total_norm.detach().cpu()


def grad_norm(
    module: Module, norm_type: Union[float, int, str], group_separator: str = "/"
) -> tuple[dict[str, Tensor], Tensor | None]:
    """Compute each parameter's gradient's norm and their overall norm.

    The overall norm is computed over all gradients together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the gradients norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the gradients viewed
            as a single vector.
        total_norm: The total p-norm of the gradients
    """
    norm_type = _validate_norm_type(norm_type)

    grads_by_device_dtype: dict[tuple[torch.device, torch.dtype], list[Tensor]] = (
        defaultdict(list)
    )
    names_by_device_dtype: dict[tuple[torch.device, torch.dtype], list[str]] = (
        defaultdict(list)
    )
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce()._values()
        key = (grad.device, grad.dtype)
        grads_by_device_dtype[key].append(grad)
        names_by_device_dtype[key].append(name)

    norms: dict[str, Tensor] = {}
    norm_groups: list[list[Tensor]] = []
    for key, grads in grads_by_device_dtype.items():
        group_norms = _foreach_norm(grads, norm_type)
        norm_groups.append(group_norms)
        for name, param_norm in zip(names_by_device_dtype[key], group_norms):
            norm_key = f"grad_{norm_type}_norm{group_separator}{name}"
            norms[norm_key] = param_norm

    total_norm = _total_norm_from_group_norms(norm_groups, norm_type)
    if total_norm is not None:
        norms[f"grad_{norm_type}_norm_total"] = total_norm
    return norms, total_norm
