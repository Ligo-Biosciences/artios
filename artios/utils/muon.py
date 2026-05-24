"""
Muon Optimizer adapted from KellerJordan

@misc{jordan2024muon,
  author       = {Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and
                  Franz Cesista and Laker Newhouse and Jeremy Bernstein},
  title        = {Muon: An optimizer for hidden layers in neural networks},
  year         = {2024},
  url          = {https://kellerjordan.github.io/posts/muon/}
}

"""

from collections import defaultdict

import torch
from torch import Tensor
from torch.optim.adam import adam

MUON_LR_SCALAR = (
    3.91918358845  # 4.0  # constant to make Muon's LR somewhat comparable to AdamW
)


def _get_scalar_dtype(is_fused=None):
    if is_fused:
        return torch.float32
    return (
        torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
    )


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def batched_zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """
    Batched gradient orthogonalization routine.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Args:
        G (Tensor): batched gradient tensor
            [B, fan_out, fan_in]
        steps (int): number of steps
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-1, -2), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = torch.bmm(X, X.mT)
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def batched_muon_update(
    grad, momentum, beta: float = 0.95, ns_steps: int = 5, nesterov: bool = True
):
    """
    Batched version of muon_update that processes multiple gradients simultaneously.

    Args:
        grad (Tensor): Batched gradient tensor [B, fan_out, fan_in]
        momentum (Tensor): Batched momentum buffer [B, fan_out, fan_in]
        beta (float): Momentum coefficient
        ns_steps (int): Number of Newton-Schulz iterations
        nesterov (bool): Whether to use Nesterov momentum

    Returns:
        Tensor: Batched update tensor [B, fan_out, fan_in]
    """
    # Calculate update
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    # Orthogonalize gradient using batched Newton-Schulz
    update = batched_zeropower_via_newtonschulz5(update, steps=ns_steps)

    # Scale update for consistent RMS
    fan_out, fan_in = grad.shape[-2:]
    update *= MUON_LR_SCALAR * (fan_out / fan_in) ** 0.5

    return update


class BatchedMuonWithAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon"]
                )
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )
        super().__init__(param_groups, dict())

    # TODO: make a _init_group function like that of Adam

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Group parameters that are the same shape together.

        # Batched muon update for all of them (for loop over different shapes)

        # adam and muon elementwise updates with foreach.

        for group in self.param_groups:
            if group["use_muon"]:
                params = []
                grads = []
                exp_avgs = []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                        # p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    # update = muon_update(
                    #    p.grad, state["momentum_buffer"], beta=group["momentum"]
                    # )
                    # muon_updates.append(update)
                    params.append(p)
                    grads.append(p.grad)
                    exp_avgs.append(state["momentum_buffer"])

                muon_step(
                    params,
                    grads,
                    exp_avgs,
                    beta=group["momentum"],
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                )

            else:  # TODO: adamw can be done with elementwise foreach lerp
                # params: List[Tensor],
                # grads: List[Tensor],
                # exp_avgs: List[Tensor],
                # exp_avg_sqs: List[Tensor],
                # max_exp_avg_sqs: List[Tensor],
                # state_steps: List[Tensor],
                params = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                max_exp_avg_sqs = []
                state_steps = []

                for p in group["params"]:
                    if p.grad is None:
                        continue
                        # p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=True),
                            device=p.device,
                        )
                    params.append(p)
                    grads.append(p.grad)
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    max_exp_avg_sqs.append(
                        torch.tensor([], dtype=p.dtype, device=p.device)
                    )
                    state_steps.append(state["step"])

                # Adam Update
                adam(
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    max_exp_avg_sqs=max_exp_avg_sqs,
                    state_steps=state_steps,
                    fused=False,
                    decoupled_weight_decay=True,
                    amsgrad=False,
                    beta1=group["betas"][0],
                    beta2=group["betas"][1],
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=1e-10,
                    maximize=False,
                )

        return loss


def muon_step(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    *,
    beta: float,
    lr: float,
    weight_decay: float,
):
    # TODO: asserts
    assert isinstance(beta, float)
    assert isinstance(lr, float)
    assert isinstance(weight_decay, float)
    # TODO: absorb the muon update function into this
    # TODO: remove the double momentum lerp

    # Compute Muon updates with batching
    grads_moments_by_shape = defaultdict(list)
    for param, grad, momentum in zip(params, grads, exp_avgs):
        assert grad.shape == momentum.shape
        shape = grad.shape
        if len(shape) == 2:  # Only process 2D parameters with Muon
            grads_moments_by_shape[shape].append((param, grad, momentum))
        elif len(shape) > 2:
            raise ValueError(f"Got parameter with rank > 2: {shape}")

    # Compute batched Muon updates
    params = []
    muon_updates = []
    for shape, params_grads_moments in grads_moments_by_shape.items():
        params_with_shape = [pgm[0] for pgm in params_grads_moments]
        grads_with_shape = [pgm[1] for pgm in params_grads_moments]
        moments_with_shape = [pgm[2] for pgm in params_grads_moments]

        batched_grads = torch.stack(grads_with_shape, dim=0)
        batched_momentum = torch.stack(moments_with_shape, dim=0)

        updates = batched_muon_update(
            grad=batched_grads,
            momentum=batched_momentum,
            beta=beta,
            ns_steps=5,
            nesterov=True,
        )

        # Unstack updates and append individual tensors
        for update in torch.unbind(updates, dim=0):
            muon_updates.append(update)

        # Re-order pointers
        for p in params_with_shape:
            params.append(p)

    # Update momentum
    torch._foreach_lerp_(exp_avgs, grads, 1 - beta)

    # Decoupled weight decay
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    # Update weights
    torch._foreach_mul_(muon_updates, -lr)
    torch._foreach_add_(params, muon_updates)
