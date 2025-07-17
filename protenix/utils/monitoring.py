from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
from torch.nn import Module


def plot_loss_vs_timestep(
    timesteps: np.ndarray,
    losses: np.ndarray,
    num_points: int = 1000,
):
    """
    Create a smooth plot of loss expectation vs timesteps.

    Args:
        timesteps (np.ndarray): timestep values in range [0, 1]
            [..., N_sample]
        losses (np.ndarray): loss values
            [..., N_sample]
        num_points (int): number of points to evaluate the smooth curve at
    """
    # Flatten arrays
    timesteps = timesteps.flatten()
    losses = losses.flatten()

    # Sort by timesteps for cleaner visualization
    sort_idx = np.argsort(timesteps)
    timesteps = timesteps[sort_idx]
    losses = losses[sort_idx]

    # Create evaluation points (linear-spaced in [0,1])
    eval_points = np.linspace(timesteps.min(), timesteps.max(), num_points)

    # Compute weighted average using gaussian kernel
    kernel = gaussian_kde(timesteps, weights=losses)
    density = gaussian_kde(timesteps)

    # Compute smoothed expectation
    expected_loss = kernel(eval_points) / (density(eval_points) + 1e-8)

    # Compute standard deviation in local neighborhoods
    std_devs = []
    window_size = (timesteps.max() - timesteps.min()) / 50  # Fixed window size
    for t in eval_points:
        # Find losses in local neighborhood using fixed window
        mask = np.abs(timesteps - t) < window_size
        if np.sum(mask) > 0:
            std_devs.append(np.std(losses[mask]))
        else:
            std_devs.append(0)
    std_devs = np.array(std_devs)

    return eval_points, expected_loss, std_devs


def visualize_loss_histogram(
    timesteps: np.ndarray,
    losses: np.ndarray,
    n_time_bins: int = 50,
    title: str = "Loss Distribution over Timesteps",
    figsize: tuple = (6, 4),
):
    """
    Create a histogram-based visualization of loss distribution over time.

    Args:
        timesteps (np.ndarray): timestep values in range [0, 1]
        losses (np.ndarray): corresponding loss values
        n_time_bins (int): number of timestep bins to divide the data into
        title (str): plot title
        figsize (tuple): figure dimensions (width, height)
    """
    # Create time bins and compute statistics
    bins = np.linspace(timesteps.min(), timesteps.max(), n_time_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    means = []
    stds = []
    for i in range(len(bins) - 1):
        mask = (timesteps >= bins[i]) & (timesteps < bins[i + 1])
        bin_losses = losses[mask]
        means.append(np.mean(bin_losses) if len(bin_losses) > 0 else 0)
        stds.append(np.std(bin_losses) if len(bin_losses) > 0 else 0)

    means = np.array(means)
    stds = np.array(stds)

    # Create figure with higher DPI for better resolution
    plt.figure(figsize=figsize, dpi=300)

    # Plot raw data points with low alpha
    plt.scatter(timesteps, losses, alpha=0.1, color="gray", label="Raw samples", s=5)

    # Plot mean values as a smooth line using interpolation with more points
    smooth_t = np.linspace(bin_centers.min(), bin_centers.max(), 500)
    smooth_means = np.interp(smooth_t, bin_centers, means)
    plt.plot(smooth_t, smooth_means, "b-", label="Mean loss", linewidth=1.5)

    # Add confidence bands (±1 standard deviation) as steps
    plt.fill_between(
        bin_centers,
        means - stds,
        means + stds,
        alpha=0.2,
        color="blue",
        label="±1 std dev",
        step="mid",
    )

    plt.xlabel("Timestep")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def grad_norm(
    module: Module, norm_type: Union[float, int, str], group_separator: str = "/"
) -> dict[str, float]:
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
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(
            f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}"
        )

    norms = {
        f"grad_{norm_type}_norm{group_separator}{name}": p.grad.data.norm(norm_type)
        for name, p in module.named_parameters()
        if p.grad is not None
    }
    total_norm = None
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type)
        norms[f"grad_{norm_type}_norm_total"] = total_norm
    return norms, total_norm
