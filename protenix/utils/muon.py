"""
Muon Optimizer adapted from KellerJordan

@misc{jordan2024muon,
  author       = {Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and
                  Franz Cesista and Laker Newhouse and Jeremy Bernstein},
  title        = {Muon: An optimizer for hidden layers in neural networks},
  year         = {2024},
  url          = {https://kellerjordan.github.io/posts/muon/}
}

TODO: if this gives performance improvements, refactor it nicely.

# TODO: there are crazy performance gains on the table by splitting the parameters with the same shapes into groups
# and doing batched processing on the GPU. Since neural nets have blocks with the same shapes, it should be simple enough
# to do this.
"""

import torch
import torch.distributed as dist

MUON_LR_SCALAR = (
    3.91918358845  # 4.0  # constant to make Muon's LR somewhat comparable to AdamW
)


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
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


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    # Calculate update
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)

    # Orthogonalize gradient
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    # scale update for consistent RMS
    fan_out, fan_in = grad.size()
    update *= MUON_LR_SCALAR * (fan_out / fan_in) ** 0.5

    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.

    NOTE: this method distributes the overhead of Muon across GPUs - instead of each GPU orthogonalizing
    the gradients redundantly, each is assigned a subset of the weights. This method is untested in highly
    distributed regimes (>32 GPUs) and relatively small models, at which point the communication overhead might
    not be worth it.


    Arguments:
        lr: The learning rate.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """

    # TODO: type hints
    def __init__(self, params, lr=1e-3, weight_decay=0.0, momentum=0.95):
        """
        Args:
            lr (float): the learning rate
            weight_decay (float): The AdamW-style weight decay.
            momentum (float): The momentum. Defaults to 0.95.
        """
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                dist.get_world_size() - len(params) % dist.get_world_size()
            )
            # Assign each param to a certain rank
            for base_i in range(len(params))[:: dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    # Each GPU orthogonalizes its parameters
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"]
                    )

                    # apply weight decay
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                    # apply update
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])

                # Sync parameters after GPUs have updated their subsets
                dist.all_gather(
                    params_pad[base_i : base_i + dist.get_world_size()],
                    params_pad[base_i + dist.get_rank()],
                )

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # TODO: separate the Muon update completely into its separate method
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(
                    p.grad, state["momentum_buffer"], beta=group["momentum"]
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


# TODO: type hints
def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
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

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = (
                    params
                    + [torch.empty_like(params[-1])]
                    * (  # empty tensors like the shape of the last parameter?
                        dist.get_world_size()
                        - len(params)
                        % dist.get_world_size()  # add the remainder from e.g. 8 GPUs - (remainder) from parameters
                    )
                )
                for base_i in range(len(params))[:: dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(
                        params
                    ):  # another remainder check?
                        # Each rank processes its own parameter
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(
                            p.grad, state["momentum_buffer"], beta=group["momentum"]
                        )
                        # apply weight decay
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(
                            update.reshape(p.shape), alpha=-group["lr"]
                        )  # apply update
                    # Share params after each rank processes its param.
                    dist.all_gather(
                        params_pad[base_i : base_i + dist.get_world_size()],
                        params_pad[base_i + dist.get_rank()],
                    )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    # apply weight decay
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])  # apply update

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
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

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
