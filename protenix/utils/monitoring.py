from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import gaussian_kde, probplot
from sklearn.manifold import TSNE
from torch.nn import Module

from protenix.common.constants import STD_RESIDUES_AND_GAP


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


def analyse_latents(latents: dict, plot_per_batch: bool = False):
    """
    Analyse the latents. These are the plots we will make:
    1. T-SNE
    2. Per dimension histogram
    3. Per dimensions QQ plot
    4. Covariance heatmap
    5. Eigenvalues of covariance matrix

    Args:
        latents (dict): A dictionary containing the latents.
          Keys: "mean", "logvar"
            mean (list[np.array]): List of mean latents tensors. Each array
                is of shape [N_sample*N_token, c_latent]
            logvar (list[np.array]): List of logvar latents tensors. Each array
                is of shape [N_sample*N_token, c_latent]
            restype (list[np.array]): List of residue labels tensors. Each array
                is of shape [N_sample*N_token]
        plot_per_batch (bool): Whether to plot per batch. If false, we combine all tokens
                across all batches and samples and plot from there.

    Returns:
        figs (dict): A dictionary containing the figures.
            Keys: "tsne", "hist", "qq", "cov_heatmap", "eigs"
            Values: plt.Figure
    """
    if not latents["mean"]:
        return {}

    # Reshape data across batch dimension
    latents["mean"] = [a.reshape(-1, a.shape[-1]) for a in latents["mean"]]
    latents["logvar"] = [a.reshape(-1, a.shape[-1]) for a in latents["logvar"]]
    latents["restype"] = [a.reshape(-1) for a in latents["restype"]]

    # Combine batches if not plotting per batch
    if not plot_per_batch:
        latents = {
            "mean": [np.concatenate(latents["mean"], axis=-2)],
            "logvar": [np.concatenate(latents["logvar"], axis=-2)],
            "restype": [np.concatenate(latents["restype"], axis=-1)],
        }

    figs = {}
    for batch_idx, (latents_mean, latents_logvar, latents_restype) in enumerate(
        zip(latents["mean"], latents["logvar"], latents["restype"])
    ):
        eps = np.random.randn(*latents_mean.shape)
        latents_sample = latents_mean + np.exp(0.5 * latents_logvar) * eps

        suffix = f"_batch_{batch_idx}" if plot_per_batch else ""

        # Plots figs
        figs[f"tsne{suffix}"] = plot_tsne(latents_sample, latents_restype, suffix)
        figs[f"hist{suffix}"] = plot_histograms(latents_sample, suffix)
        figs[f"qq{suffix}"] = plot_qq_plots(latents_sample, suffix)
        figs[f"cov_heatmap{suffix}"] = plot_covariance_heatmap(latents_sample, suffix)
        figs[f"eigs{suffix}"] = plot_eigenvalues(latents_sample, suffix)

        # Additional histograms for mean and logvar
        figs[f"hist_mean{suffix}"] = plot_histograms(latents_mean, suffix)
        figs[f"hist_logvar{suffix}"] = plot_histograms(latents_logvar, suffix)

    return figs


def plot_tsne(
    data: np.ndarray, labels: np.ndarray, batch_suffix: str = ""
) -> plt.Figure:
    """Create T-SNE visualization of latent means and return the figure."""
    tsne_data = TSNE(
        n_components=2,
        perplexity=10,
        max_iter=1000,
        init="random",
    ).fit_transform(data)

    # Create reverse mapping from index to residue name
    index_to_residue = {v: k for k, v in STD_RESIDUES_AND_GAP.items()}

    # unique labels
    uniq = np.unique(labels)
    # keep a stable ordering
    uniq.sort()
    colors = distinct_colors(len(uniq))

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, label_idx in enumerate(uniq):
        mask = labels == label_idx
        # Map index to actual residue name
        residue_name = index_to_residue.get(label_idx, f"UNK_{label_idx}")
        ax.scatter(
            tsne_data[mask, 0],
            tsne_data[mask, 1],
            color=colors[idx],
            label=residue_name,
        )

    ax.set_title(f"T-SNE visualization of latents{batch_suffix}")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        fontsize="small",
        markerscale=2,
    )

    fig.tight_layout()
    return fig


def plot_histograms(data: np.ndarray, batch_suffix: str = "") -> plt.Figure:
    """Create histograms for each latent dimension and return the figure."""
    n_dims = data.shape[-1]
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    # Normalise axes array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_dims):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        ax.hist(data[:, i], bins=50, alpha=0.7, edgecolor="black")
        ax.set_title(f"Latent dim {i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for i in range(n_dims, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Histograms of latent dimensions{batch_suffix}")
    fig.tight_layout()
    return fig


def plot_qq_plots(data: np.ndarray, batch_suffix: str = "") -> plt.Figure:
    """Create QQ plots comparing each dimension to a normal distribution and return the figure."""
    n_dims = data.shape[-1]
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    # Normalise axes array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_dims):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        probplot(data[:, i], dist="norm", plot=ax)
        ax.set_title(f"QQ plot - dim {i}")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for i in range(n_dims, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"QQ plots vs Normal distribution{batch_suffix}")
    fig.tight_layout()
    return fig


def plot_covariance_heatmap(data: np.ndarray, batch_suffix: str = "") -> plt.Figure:
    """Create covariance matrix heatmap and return the figure."""
    cov_matrix = np.cov(data.T)
    n_dims = data.shape[-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cov_matrix,
        annot=(n_dims <= 20),
        cmap="RdBu_r",
        center=0,
        square=True,
        fmt=".3f" if n_dims <= 20 else None,
        ax=ax,
    )
    ax.set_title(f"Covariance matrix of latent dimensions{batch_suffix}")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Latent Dimension")
    fig.tight_layout()
    return fig


def plot_eigenvalues(data: np.ndarray, batch_suffix: str = "") -> plt.Figure:
    """Create eigenvalue plot of covariance matrix and return the figure."""
    cov_matrix = np.cov(data.T)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        range(1, len(eigenvalues) + 1),
        eigenvalues,
        "o-",
        markersize=4,
        label="Eigenvalues",
    )
    ax1.set_title(f"Eigenvalues of covariance matrix{batch_suffix}")
    ax1.set_xlabel("Eigenvalue Index")
    ax1.set_ylabel("Eigenvalue")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.plot(
        range(1, len(eigenvalues) + 1),
        cumvar,
        "r--",
        alpha=0.7,
        label="Cumulative variance",
    )
    ax2.set_ylabel("Cumulative Variance", color="r")
    ax2.set_ylim(0, 1)
    # Make sure legend shows
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    return fig


def distinct_colors(n: int) -> np.ndarray:
    """Return n visually distinct RGBA colors using tab20/20b/20c."""
    tabs = [
        mpl.colormaps["tab20"].colors,
        mpl.colormaps["tab20b"].colors,
        mpl.colormaps["tab20c"].colors,
    ]
    full = np.vstack(tabs)  # 60 colors total
    if n > len(full):
        raise ValueError(f"Need {n} colors, only {len(full)} pre-defined.")
    return np.array(full[:n])
