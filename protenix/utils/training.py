# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os

import torch
import torch.distributed as dist
import wandb


def get_adamw(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer for the given model with specified parameters.

    Args:
        model (torch.nn.Module): The model for which the optimizer is created.
        weight_decay (float): The weight decay (L2 penalty) for the optimizer.
        learning_rate (float): The learning rate for the optimizer.
        betas (tuple): Coefficients used for computing running averages of gradient and its square.
        device_type (str): The device type ('cuda' or 'cpu') on which the optimizer will operate.

    Returns:
        torch.optim.AdamW: The AdamW optimizer configured with the specified parameters.
    """
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    print(f"using fused AdamW: {use_fused}")

    return optimizer


def get_optimizer(configs, model: torch.nn.Module) -> torch.optim.Optimizer:
    if configs.adam.use_adamw:
        optimizer = get_adamw(
            model=model,
            weight_decay=configs.adam.weight_decay,
            learning_rate=configs.adam.lr,
            betas=(configs.adam.beta1, configs.adam.beta2),
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.adam.lr,
            weight_decay=configs.adam.weight_decay,
            betas=(configs.adam.beta1, configs.adam.beta2),
        )
    return optimizer


def is_loss_nan_check(loss: torch.Tensor) -> bool:
    """check the validness of the current loss

    Args:
        loss: the loss from the model

    Returns:
        bool: if True, loss is not nan or inf
    """

    def is_nan(x):
        return torch.isnan(x).any() or torch.isinf(x).any()

    def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM):
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor

    nan_flag = torch.tensor(
        1.0 if is_nan(loss) else 0.0,
        device=loss.device if torch.cuda.is_available() else None,
    )  # support cpu
    # avoid "Watchdog caught collective operation timeout" error
    all_reduce_tensor(nan_flag)
    if nan_flag.item() > 0.0:
        return True
    return False


def extract_gamma_step(checkpoint_path: str) -> tuple[int, float]:
    """
    Extract step and gamma values from checkpoint path.
    
    Args:
        checkpoint_path: Path string of form "{checkpoint_dir}/{step}ema{gamma}.pt"
        
    Returns:
        Tuple of (step, gamma)
        
    Example:
        >>> extract_gamma_step("/path/to/checkpoints/1000ema6.94.pt")
        (1000, 6.94)
    """
    try:
        # Get filename without directory and extension
        filename = os.path.basename(checkpoint_path)
        filename = os.path.splitext(filename)[0]  # Remove .pt
        
        # Split on "ema"
        step_str, gamma_str = filename.split("ema")
        
        # Convert to appropriate types
        step = int(step_str)
        gamma = float(gamma_str)
        
        return gamma, step

    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid checkpoint path format: {checkpoint_path}. Expected format: '{{step}}ema{{gamma}}.pt'") from e
    

def log_validation_curves(metrics_dict, step):
    """
    Logs one WandB plot per loss type, with multiple lines for each EMA variant.
    
    :param metrics_dict: A dictionary of the form:
        {
            'train_ema6.94_': {
                'weighted_smooth_lddt_loss.avg': 3.2192340016365053,
                'mse_loss.avg': 0.9385715276002884,
                'loss.avg': 6.973520064353943,
                ...
            },
            'train_ema16.97_': {
                'weighted_smooth_lddt_loss.avg': 3.2052199482917785,
                'mse_loss.avg': 0.9485592111945153,
                'loss.avg': 6.999456799030304,
                ...
            },
            ...
        }
    :param step: The current validation step (integer) or epoch number.
    """
    # Make sure there is at least one EMA variant in the dictionary
    if not metrics_dict:
        return
    
    # We'll assume all EMA sub-dicts share the same set of keys (loss types).
    ema_names = list(metrics_dict.keys()) # e.g. ["train_ema6.94_", "train_ema16.97_", ...]
    loss_keys = list(metrics_dict[ema_names[0]].keys())  # e.g. ["mse_loss.avg", "loss.avg", ...]
    
    # For each loss key, we log exactly one plot
    for loss_key in loss_keys:
        # Create a table with columns: [step, ema, metric_value]
        table = wandb.Table(columns=["step", "ema", loss_key])
        
        # Populate the table rows
        for ema_name in ema_names:
            metric_value = metrics_dict[ema_name][loss_key]
            table.add_data(step, ema_name, metric_value)
        
        # Create a multi-line plot using stroke='ema'
        line_plot = wandb.plot.line(
            table, 
            x="step", 
            y=loss_key, 
            stroke="ema", 
            title=f"{loss_key}"
        )
        
        # Log the plot under a key that indicates which loss metric it corresponds to
        wandb.log({f"{loss_key}_plot": line_plot}, step=step)