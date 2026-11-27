import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor

from artios.model.modules.penalty import PenaltyCollection
from artios.model.modules.potts import (
    compute_potts_energy,
    compute_potts_single_logits,
)
from artios.utils.geometry import centre_random_augmentation
from artios.utils.typing import Bool, Float, Int, typecheck

logger = logging.getLogger(__name__)

# EDM coordinate standard deviation used by AF3/Protenix preconditioning.
SIGMA_DATA = 16.0


class TrainingTimestepSampler(ABC):
    """Abstract base class for training timestep samplers."""

    @abstractmethod
    @typecheck
    def sample(
        self,
        size: tuple[int, ...],
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "..."]:
        """Sample timesteps for training.

        Args:
            size: Shape of output tensor.
            device: Device to place tensor on.
            generator: Generator for reproducibility.

        Returns:
            Sampled timesteps tensor.
        """
        ...


class InferenceTimestepScheduler(ABC):
    """Abstract base class for inference timestep schedulers."""

    @abstractmethod
    @typecheck
    def get_schedule(
        self,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Float[Tensor, "n_steps_plus_one"]:
        """Generate timesteps for inference.

        Args:
            n_steps: Number of timesteps.
            device: Device to place tensor on.
            dtype: Data type of tensor.

        Returns:
            Timesteps tensor of shape [n_steps + 1].
        """
        ...


class EDMTrainingNoiseSampler(TrainingTimestepSampler):
    """Sample EDM training noise levels from the AF3/Protenix lognormal law."""

    @typecheck
    def sample(
        self,
        size: tuple[int, ...],
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "..."]:
        p_mean, p_std = -1.2, 1.5
        rnd_normal = torch.randn(size=size, device=device, generator=generator)
        return torch.exp(rnd_normal * p_std + p_mean) * SIGMA_DATA


class EDMInferenceNoiseScheduler(InferenceTimestepScheduler):
    """Karras/AF3 EDM inference noise scheduler.

    Produces coordinate-space noise levels scaled by ``SIGMA_DATA``.
    """

    @typecheck
    def get_schedule(
        self,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Float[Tensor, "n_steps_plus_one"]:
        s_max, s_min, rho = 160.0, 4e-4, 7.0
        step_indices = torch.arange(n_steps + 1, dtype=dtype, device=device)
        step_size = 1 / n_steps
        schedule = (
            SIGMA_DATA
            * (
                s_max ** (1 / rho)
                + step_indices * step_size * (s_min ** (1 / rho) - s_max ** (1 / rho))
            )
            ** rho
        )
        schedule[-1] = 0
        return schedule


# =============================================================================
# Stochasticity Schedules for SDE Integration
# =============================================================================
#
# These define g(t) in the SDE: dx = v*dt + g(t)*s*dt + sqrt(2*g(t)*gamma)*dW
# where v is velocity, s is score, and gamma is temperature.


# =============================================================================
# Classifier-Free Guidance Schedules
# =============================================================================
#
# These define w(t) for CFG: v_cfg = v_uncond + w(t) * (v_cond - v_uncond)
# When w(t) = 1, no guidance is applied. When w(t) > 1, conditioning is amplified.


# =============================================================================
# Atomic Integration Primitives
# =============================================================================


@runtime_checkable
class EDMDenoisePredictor(Protocol):
    """Protocol for EDM models that predict denoised coordinates."""

    @typecheck
    def __call__(
        self,
        x_noisy: Float[Tensor, "*batch n_atom 3"],
        noise_level: Float[Tensor, "*batch"],
    ) -> Float[Tensor, "*batch n_atom 3"]:
        """Predict denoised coordinates at a given EDM noise level."""
        ...


@dataclass
class SamplingResult:
    """Result of flow matching sampling, optionally including trajectory.

    Attributes:
        final: Final denoised coordinates after all integration steps.
        trajectory: Optional trajectory of coordinates at each step.
            Shape [..., n_steps+1, n_atom, 3] where index 0 is initial noise
            and index -1 is final sample. None if save_trajectory=False.
    """

    final: Float[Tensor, "... n_atom 3"]
    trajectory: Float[Tensor, "... n_steps_plus_one n_atom 3"] | None = None


@dataclass(frozen=True)
class EDMSampler:
    """AF3-style EDM predictor-corrector sampler."""

    noise_scheduler: InferenceTimestepScheduler

    @typecheck
    def sample(
        self,
        predictor: EDMDenoisePredictor,
        x_init: Float[Tensor, "*batch n_atom 3"],
        n_steps: int,
        mask: Bool[Tensor, "*batch n_atom"],
        track_trajectory: bool = False,
    ) -> SamplingResult:
        gamma0, gamma_min, noise_scale_lambda, step_scale_eta = 0.8, 1.0, 1.003, 1.5

        device = x_init.device
        dtype = x_init.dtype
        noise_schedule = self.noise_scheduler.get_schedule(n_steps, device, dtype)

        if track_trajectory:
            traj_shape = (*x_init.shape[:-2], n_steps + 1, *x_init.shape[-2:])
            trajectory = torch.zeros(traj_shape, dtype=dtype, device=device)
            trajectory[..., 0, :, :] = x_init.detach()
        else:
            trajectory = None

        x = x_init
        for i, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            x = (
                centre_random_augmentation(x_input_coords=x, n_sample=1, mask=mask)
                .squeeze(dim=-3)
                .to(dtype)
            )

            gamma = gamma0 if c_tau > gamma_min else 0.0
            t_hat_scalar = c_tau_last * (1.0 + gamma)
            delta_noise_level = torch.sqrt(t_hat_scalar**2 - c_tau_last**2)
            x_noisy = x + noise_scale_lambda * delta_noise_level * torch.randn_like(x)

            t_hat = t_hat_scalar.reshape((1,) * len(x.shape[:-2])).expand(x.shape[:-2])
            t_hat = t_hat.to(dtype=dtype)

            x_denoised = predictor(x_noisy, t_hat)
            delta = (x_noisy - x_denoised) / t_hat[..., None, None]
            dt = c_tau - t_hat
            x = x_noisy + step_scale_eta * dt[..., None, None] * delta

            if track_trajectory:
                trajectory[..., i + 1, :, :] = x.detach()

        return SamplingResult(final=x, trajectory=trajectory)


class PottsProposal(Enum):
    """Proposal method for Potts sampling."""

    GIBBS = "gibbs"
    DLMC = "dlmc"


@dataclass
class PottsSamplingResult:
    """Result of Potts MCMC sampling."""

    sequence: Float[Tensor, "batch n_token num_residue_types"]
    energy: Float[Tensor, "batch n_token"]
    sequence_trajectory: Int[Tensor, "batch n_saved n_token"] | None = None
    energy_trajectory: Float[Tensor, "batch n_saved n_token"] | None = None
    trajectory_sweep_indices: list[int] | None = None
    u_i_trajectory: (
        Float[Tensor, "batch n_saved_ui n_token num_residue_types"] | None
    ) = None
    u_i_sweep_indices: list[int] | None = None
    penalty_grad_trajectories: (
        dict[str, Float[Tensor, "batch n_saved_ui n_token num_residue_types"]] | None
    ) = None
    penalty_names: list[str] | None = None


def _color_graph(
    edge_index: Int[Tensor, "batch n_token neighbours"],
    edge_mask: Bool[Tensor, "batch n_token neighbours"],
    *,
    active_site_mask: Bool[Tensor, "batch n_token"] | None = None,
    granular_active_site_coloring: bool = False,
) -> list[Bool[Tensor, "batch n_token"]]:
    """Greedy graph colouring for parallel Gibbs updates.

    Assigns each node a colour different from those of the neighbours in its
    own list that were coloured before it. On a directed k-NN graph a pair
    listed in one direction only can still share a colour (about 1% of edges
    on typical proteins), so a sweep is approximately, not exactly, Gibbs.
    Each batch element is coloured on its own graph, so a batch may mix
    backbones; elements with identical graphs get identical colourings.

    If `granular_active_site_coloring` is True and `active_site_mask` is
    provided, each active-site token is pre-assigned a unique colour
    (0..n_active-1) so no two active-site tokens can share a colour group.
    Non-active-site tokens are then greedy-coloured starting from colour 0,
    and may freely reuse those reserved colours where the standard graph-
    colouring constraint allows it (no edge to a same-colour neighbour).

    Returns:
        List of boolean masks, one per colour (empty for batch elements that
        use fewer colours).
    """
    colorings = [
        _greedy_coloring(
            edge_index[b].tolist(),
            edge_mask[b].tolist(),
            torch.where(active_site_mask[b])[0].tolist()
            if granular_active_site_coloring and active_site_mask is not None
            else [],
        )
        for b in range(edge_index.shape[0])
    ]
    colors = torch.tensor(colorings, device=edge_index.device)  # [B, N]
    num_colors = int(colors.max()) + 1
    return [colors == c for c in range(num_colors)]


def _greedy_coloring(
    edge_index: list[list[int]], edge_mask: list[list[bool]], reserved: list[int]
) -> list[int]:
    """Greedy colouring of one graph; ``reserved`` nodes get colours 0, 1, ..."""
    colors = [-1] * len(edge_index)
    for c, i in enumerate(reserved):
        colors[i] = c
    for i, (neighbours, valid) in enumerate(zip(edge_index, edge_mask)):
        if colors[i] >= 0:
            continue  # already coloured (active site)
        taken = {colors[j] for j, ok in zip(neighbours, valid) if ok and j != i}
        c = 0
        while c in taken:
            c += 1
        colors[i] = c
    return colors


@dataclass
class ProposalResult:
    """Result of a Potts proposal step (Gibbs or DLMC)."""

    S_new: Float[Tensor, "batch n_token num_residue_types"]
    energy: Float[Tensor, "batch"] | None = None
    log_probs: Float[Tensor, "batch n_token num_residue_types"] | None = None


def _potts_proposal_gibbs(
    S: Float[Tensor, "batch n_token num_residue_types"],
    h: Float[Tensor, "batch n_token num_residue_types"],
    J: Float[Tensor, "batch n_token neighbours num_residue_types num_residue_types"],
    edge_index: Int[Tensor, "batch n_token neighbours"],
    edge_mask: Bool[Tensor, "batch n_token neighbours"],
    token_mask: Bool[Tensor, "batch n_token"] | None,
    temperature: float,
    color_masks: list[Bool[Tensor, "batch n_token"]],
    generator: torch.Generator | None = None,
    penalty: Callable[[Tensor], Tensor] | None = None,
    h_uncond=None,
    J_uncond=None,
    edge_index_uncond=None,
    edge_mask_uncond=None,
    cfg_scale: float = 0.0,
    active_site_mask: Bool[Tensor, "batch n_token"] | None = None,
    active_site_only_cfg: bool = False,
) -> ProposalResult:
    """Gibbs proposal with parallel colour sweeps.

    For each colour group (independent set), computes the per-position
    sampling distribution, samples new states, and updates S at those
    positions.

    If penalty is provided, its gradient w.r.t. S is computed via autograd
    and added to the per-position energies of *both* the conditional and
    unconditional pathways before softmaxing (following Chroma). This way
    the penalty acts as a shared bias regardless of CFG state.

    If unconditional parameters are provided with cfg_scale != 1.0,
    classifier-free guidance is applied by blending conditional and
    unconditional distributions in probability space:

        p_cfg(a) = p_uncond(a) + cfg_scale * (p_cond(a) - p_uncond(a))

    Negative entries (which can arise for cfg_scale > 1 or < 0) are
    clamped to zero and the row is renormalised. Reduces to p_cond at
    cfg_scale=1 and to p_uncond at cfg_scale=0.
    """
    R = S.shape[-1]
    use_cfg = h_uncond is not None and cfg_scale != 1.0

    for color_mask in color_masks:
        S_idx = S.argmax(-1)  # [B, N]

        # Potts conditional energy: U_i(a) = h_i(a) + sum_j J_ij(a, s_j)
        logits = compute_potts_single_logits(S_idx, h, J, edge_index, edge_mask)
        U_i = -logits  # [B, N, R]

        # Compute penalty gradient once. We add it to *both* the conditional
        # and unconditional energies below so the penalty applies as a
        # shared bias on both pathways.
        U_i_adjustment = None
        if penalty is not None:
            with torch.enable_grad():
                S_grad = S.detach().requires_grad_(True)
                U_penalty = penalty(S_grad)
                U_i_adjustment = torch.autograd.grad(U_penalty.sum(), S_grad)[
                    0
                ].detach()
            U_i = U_i + 0.5 * U_i_adjustment

        probs = F.softmax(-U_i / temperature, dim=-1)  # [B, N, R]

        # Classifier-free guidance: blend conditional and unconditional
        # distributions in probability space.
        if use_cfg:
            logits_uncond = compute_potts_single_logits(
                S_idx, h_uncond, J_uncond, edge_index_uncond, edge_mask_uncond
            )
            U_i_uncond = -logits_uncond
            if U_i_adjustment is not None:
                U_i_uncond = U_i_uncond + 0.5 * U_i_adjustment
            probs_uncond = F.softmax(-U_i_uncond / temperature, dim=-1)
            probs_cfg = probs_uncond + cfg_scale * (probs - probs_uncond)
            probs_cfg = probs_cfg.clamp(min=0.0)
            probs_cfg = probs_cfg / probs_cfg.sum(dim=-1, keepdim=True).clamp(min=1e-30)
            if active_site_only_cfg and active_site_mask is not None:
                probs = torch.where(active_site_mask.unsqueeze(-1), probs_cfg, probs)
            else:
                probs = probs_cfg

        B, N, _ = probs.shape
        flat_probs = probs.reshape(B * N, R)
        samples = torch.multinomial(flat_probs, 1, generator=generator).reshape(B, N)
        S_new = F.one_hot(samples, R).float()

        update_mask = color_mask
        if token_mask is not None:
            update_mask = update_mask & token_mask
        S = torch.where(update_mask.unsqueeze(-1), S_new, S)

    return ProposalResult(S_new=S)


def _dlmc_energy_and_log_probs(
    S,
    h,
    J,
    edge_index,
    edge_mask,
    token_mask,
    temperature,
    dt=0.1,
    penalty=None,
    h_uncond=None,
    J_uncond=None,
    edge_index_uncond=None,
    edge_mask_uncond=None,
    cfg_scale: float = 0.0,
    active_site_mask: Bool[Tensor, "batch n_token"] | None = None,
    active_site_only_cfg: bool = False,
):
    """Compute DLMC energy and transition log-probabilities for a given state.

    Returns (energy, log_probs) where log_probs includes both off-diagonal
    transition probabilities and the self-transition (remaining mass).

    If unconditional parameters are provided with cfg_scale != 1.0,
    classifier-free guidance is applied to the per-position distribution
    in probability space (arithmetic mixture, clamped and renormalised),
    but NOT to total energy (used for MH acceptance).

    If active_site_only_cfg is True and active_site_mask is provided,
    the CFG blend is applied only at active-site positions; non-active
    positions use the plain conditional distribution.

    If penalty is provided, the same penalty energy bias is added to both
    the conditional and unconditional per-position energies before
    softmaxing, and the raw penalty value is added to the total energy.
    """
    S_idx = S.argmax(-1)
    use_cfg = h_uncond is not None and cfg_scale != 1.0

    # Conditional energy for all candidate states
    logits = compute_potts_single_logits(
        S_idx, h, J, edge_index, edge_mask
    )  # [B, N, R]
    E_all = -logits  # [B, N, R]

    # Compute total energy (NOT blended — used for MH acceptance).
    energy = compute_potts_energy(
        S_idx, h, J, edge_index, edge_mask, collate_nodes=True
    )

    # Compute penalty gradient once. We add it to *both* the conditional
    # and unconditional per-position energies below so the penalty applies
    # symmetrically. The raw penalty value is added to the total energy.
    U_i_adjustment = None
    if penalty is not None:
        with torch.enable_grad():
            S_grad = S.detach().requires_grad_(True)
            U_penalty = penalty(S_grad)
            U_i_adjustment = torch.autograd.grad(U_penalty.sum(), S_grad)[0].detach()
            U_i_adjustment = U_i_adjustment - torch.gather(
                U_i_adjustment, -1, S_idx.unsqueeze(-1)
            )
        E_all = E_all + U_i_adjustment
        energy = energy + U_penalty.detach()

    probs = F.softmax(-E_all / temperature, dim=-1)  # [B, N, R]

    # Classifier-free guidance: blend conditional and unconditional
    # distributions in probability space.
    if use_cfg:
        logits_uncond = compute_potts_single_logits(
            S_idx, h_uncond, J_uncond, edge_index_uncond, edge_mask_uncond
        )
        E_all_uncond = -logits_uncond
        if U_i_adjustment is not None:
            E_all_uncond = E_all_uncond + U_i_adjustment
        probs_uncond = F.softmax(-E_all_uncond / temperature, dim=-1)
        probs_cfg = probs_uncond + cfg_scale * (probs - probs_uncond)
        probs_cfg = probs_cfg.clamp(min=0.0)
        probs_cfg = probs_cfg / probs_cfg.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        if active_site_only_cfg and active_site_mask is not None:
            # Apply CFG only at active-site positions; leave the rest as
            # plain conditional probs.
            probs = torch.where(active_site_mask.unsqueeze(-1), probs_cfg, probs)
        else:
            probs = probs_cfg

    # Convert per-position probs to log probs for the DLMC transition rate
    # computation below.
    logP_j = probs.clamp(min=1e-30).log()  # [B, N, R]

    # Get log probs of current state
    logP_i = torch.gather(logP_j, -1, S_idx.unsqueeze(-1))  # [B, N, 1]

    ### We now compare the log probs of the potential future states to the current states
    ### If the difference is high, then we assign a high mutation rate (Q_ij).
    ### If the difference is low, then we assign a low mutation rate (Q_ij).
    # Calculate transition rates
    log_Q_ij = F.logsigmoid(logP_j - logP_i)  # [B, N, R]
    rate = torch.exp(log_Q_ij - logP_j)  # [B, N, R]

    ### The mutation rates Q_ij determine a probability of mutation over time period dt
    ### as can be calculated using properties of the Poisson process.
    # Transition probability = P_j * (1 - exp(-dt * rate))
    logP_ij = logP_j + (-(-dt * rate).expm1()).log()  # [B, N, R]

    # Remove self-transition probabilities
    logP_ij = logP_ij + (1.0 - S).clamp(min=1e-30).log()

    # Prob residue i does not mutate is 1 - sum_j P_ij
    p_flip = ((1.0 - S) * logP_ij.exp()).sum(-1, keepdim=True)
    logP_ii = (1.0 - p_flip).clamp(min=1e-5).log()
    log_probs = torch.where(S.bool(), logP_ii, logP_ij)

    return energy, log_probs


def _potts_proposal_dlmc(
    S,
    h,
    J,
    edge_index,
    edge_mask,
    token_mask,
    temperature,
    dt=0.1,
    generator=None,
    penalty=None,
    h_uncond=None,
    J_uncond=None,
    edge_index_uncond=None,
    edge_mask_uncond=None,
    cfg_scale: float = 0.0,
    active_site_mask: Bool[Tensor, "batch n_token"] | None = None,
    active_site_only_cfg: bool = False,
):
    B, N, R = S.shape

    # get log probs of all potential mutations
    energy, log_probs = _dlmc_energy_and_log_probs(
        S,
        h,
        J,
        edge_index,
        edge_mask,
        token_mask,
        temperature,
        dt,
        penalty,
        h_uncond=h_uncond,
        J_uncond=J_uncond,
        edge_index_uncond=edge_index_uncond,
        edge_mask_uncond=edge_mask_uncond,
        cfg_scale=cfg_scale,
        active_site_mask=active_site_mask,
        active_site_only_cfg=active_site_only_cfg,
    )

    # sample from the log probs to get the mutations
    flat_probs = log_probs.exp().reshape(B * N, R)
    flat_probs = flat_probs / flat_probs.sum(-1, keepdim=True)  # renormalize
    samples = torch.multinomial(flat_probs, 1, generator=generator).reshape(B, N)
    S_new = F.one_hot(samples, R).float()

    if token_mask is not None:
        S_new = torch.where(token_mask.unsqueeze(-1), S_new, S)

    return ProposalResult(S_new=S_new, energy=energy, log_probs=log_probs)


def _mh_rejection_step(
    S: Float[Tensor, "batch n_token num_residue_types"],
    forward_result: ProposalResult,
    h,
    J,
    edge_index,
    edge_mask,
    token_mask,
    temperature: float,
    dt: float,
    penalty: Callable[[Tensor], Tensor] | None,
    generator: torch.Generator | None,
    h_uncond=None,
    J_uncond=None,
    edge_index_uncond=None,
    edge_mask_uncond=None,
    cfg_scale: float = 0.0,
    active_site_mask: Bool[Tensor, "batch n_token"] | None = None,
    active_site_only_cfg: bool = False,
) -> Float[Tensor, "batch n_token num_residue_types"]:
    """Metropolis-Hastings correction for DLMC proposals.

    Computes the acceptance ratio by comparing the forward flux
    (S -> S_new) with the reverse flux (S_new -> S), and accepts
    the proposal with probability min(1, exp(flux_reverse - flux_forward)).
    """
    S_new = forward_result.S_new
    U_forward = forward_result.energy
    logp_forward = forward_result.log_probs

    # Compute reverse: energy and log-probs at the proposed state
    U_reverse, logp_reverse = _dlmc_energy_and_log_probs(
        S_new,
        h,
        J,
        edge_index,
        edge_mask,
        token_mask,
        temperature,
        dt,
        penalty,
        h_uncond=h_uncond,
        J_uncond=J_uncond,
        edge_index_uncond=edge_index_uncond,
        edge_mask_uncond=edge_mask_uncond,
        cfg_scale=cfg_scale,
        active_site_mask=active_site_mask,
        active_site_only_cfg=active_site_only_cfg,
    )

    # Flux = -U/T + sum_i log P(transition to target)
    # Forward flux: log-prob of transitioning S -> S_new under forward proposal
    logp_s_to_snew = torch.gather(
        logp_forward, -1, S_new.argmax(-1).unsqueeze(-1)
    ).squeeze(-1)  # [B, N]
    flux_forward = -U_forward / temperature + logp_s_to_snew.sum(-1)  # [B]

    # Reverse flux: log-prob of transitioning S_new -> S under reverse proposal
    logp_snew_to_s = torch.gather(logp_reverse, -1, S.argmax(-1).unsqueeze(-1)).squeeze(
        -1
    )  # [B, N]
    flux_reverse = -U_reverse / temperature + logp_snew_to_s.sum(-1)  # [B]

    # Acceptance ratio
    acc_ratio = torch.exp(flux_reverse - flux_forward).clamp(max=1.0)  # [B]
    accept = torch.bernoulli(acc_ratio, generator=generator)  # [B]

    return torch.where(accept[:, None, None].bool(), S_new, S)


@dataclass
class TrajectoryTracker:
    """Class for tracking mutation trajectory during Potts sampling."""

    save_set: set[int]
    save_list: list[int]
    seq_traj: Int[Tensor, "batch n_saved n_token"]
    energy_traj: Float[Tensor, "batch n_saved n_token"]
    u_i_save_set: set[int] | None = None
    u_i_save_list: list[int] | None = None
    penalty_collection: PenaltyCollection | None = None
    penalty_grad_trajs: dict | None = None
    u_i_traj: Float[Tensor, "batch n_saved_ui n_token num_residue_types"] | None = None

    @staticmethod
    def create(
        n_sweeps: int,
        interval: int,
        S: Float[Tensor, "batch n_token num_residue_types"],
        h: Float[Tensor, "batch n_token num_residue_types"],
        J: Float[
            Tensor, "batch n_token neighbours num_residue_types num_residue_types"
        ],
        edge_index: Int[Tensor, "batch n_token neighbours"],
        edge_mask: Bool[Tensor, "batch n_token neighbours"],
        u_i_interval: int | None = None,
        penalty: Callable[[Tensor], Tensor] | None = None,
    ) -> "TrajectoryTracker":
        B, N, R = S.shape

        save_list = list(range(0, n_sweeps + 1, interval))
        if n_sweeps not in save_list:
            save_list.append(n_sweeps)
        n_saved = len(save_list)

        seq_traj = torch.zeros(B, n_saved, N, dtype=torch.long, device=S.device)
        energy_traj = torch.zeros(B, n_saved, N, device=S.device)

        S_idx = S.argmax(-1)

        seq_traj[:, 0] = S_idx
        energy_traj[:, 0] = compute_potts_energy(S_idx, h, J, edge_index, edge_mask)

        # U_i tracking
        u_i_save_set = None
        u_i_save_list = None
        u_i_traj = None
        if u_i_interval is not None:
            u_i_save_list = list(range(0, n_sweeps + 1, u_i_interval))
            if n_sweeps not in u_i_save_list:
                u_i_save_list.append(n_sweeps)
            u_i_save_set = set(u_i_save_list)
            n_saved_ui = len(u_i_save_list)
            u_i_traj = torch.zeros(B, n_saved_ui, N, R, device=S.device)
            logits = compute_potts_single_logits(S_idx, h, J, edge_index, edge_mask)
            u_i_traj[:, 0] = -logits

        # Penalty gradient tracking (at same interval as U_i)
        p_collection = None
        p_grad_trajs = None
        if u_i_interval is not None and penalty is not None:
            p_collection = penalty
            p_grad_trajs = {}
            for name, fn in p_collection.penalties.items():
                traj = torch.zeros(B, n_saved_ui, N, R, device=S.device)
                # Record initial gradient at sweep 0
                with torch.enable_grad():
                    S_grad = S.detach().requires_grad_(True)
                    val = fn(S_grad)
                    grad = torch.autograd.grad(val.sum(), S_grad)[0].detach()
                traj[:, 0] = grad
                p_grad_trajs[name] = traj

        return TrajectoryTracker(
            save_set=set(save_list),
            save_list=save_list,
            seq_traj=seq_traj,
            energy_traj=energy_traj,
            u_i_save_set=u_i_save_set,
            u_i_save_list=u_i_save_list,
            u_i_traj=u_i_traj,
            penalty_collection=p_collection,
            penalty_grad_trajs=p_grad_trajs,
        )

    def record(
        self,
        sweep: int,
        S: Float[Tensor, "batch n_token num_residue_types"],
        h: Float[Tensor, "batch n_token num_residue_types"],
        J: Float[
            Tensor, "batch n_token neighbours num_residue_types num_residue_types"
        ],
        edge_index: Int[Tensor, "batch n_token neighbours"],
        edge_mask: Bool[Tensor, "batch n_token neighbours"],
    ) -> None:
        S_idx = S.argmax(-1)

        if sweep in self.save_set:
            idx = self.save_list.index(sweep)
            self.seq_traj[:, idx] = S_idx
            self.energy_traj[:, idx] = compute_potts_energy(
                S_idx, h, J, edge_index, edge_mask
            )

        if self.u_i_save_set is not None and sweep in self.u_i_save_set:
            idx = self.u_i_save_list.index(sweep)
            logits = compute_potts_single_logits(S_idx, h, J, edge_index, edge_mask)
            self.u_i_traj[:, idx] = -logits

            # Per-penalty gradient tracking
            if (
                self.penalty_collection is not None
                and self.penalty_grad_trajs is not None
            ):
                for name, fn in self.penalty_collection.penalties.items():
                    with torch.enable_grad():
                        S_grad = S.detach().requires_grad_(True)
                        val = fn(S_grad)
                        grad = torch.autograd.grad(val.sum(), S_grad)[0].detach()
                    self.penalty_grad_trajs[name][:, idx] = grad


@dataclass(frozen=True)
class PottsSampler:
    """MCMC sampler for Potts energy landscapes.

    Supports Gibbs sampling (with graph colouring for parallel updates)
    and Discrete Langevin Monte Carlo (DLMC). Uses simulated annealing
    to move from exploration to exploitation.
    """

    proposal: PottsProposal = PottsProposal.GIBBS
    """Proposal method: Gibbs or DLMC."""

    n_sweeps: int = 100
    """Number of MCMC sweeps."""

    temperature: float = 0.1
    """Final sampling temperature."""

    temperature_init: float = 1.0
    """Initial temperature for annealing."""

    annealing_fraction: float = 0.8
    """Fraction of sweeps spent annealing from temperature_init to temperature."""

    dt: float = 0.1
    """DLMC step size (only used when proposal is DLMC)."""

    rejection_step: bool = False
    """If True, apply Metropolis-Hastings correction to DLMC proposals."""

    track_trajectory: bool = False
    """If True, store sequence (as argmax indices) and energy at saved sweeps."""

    trajectory_interval: int = 1
    """Save trajectory every N sweeps. Initial and final states are always saved."""

    track_u_i: bool = False
    """If True, store per-node conditional energies U_i [B, N, R] at saved sweeps."""

    u_i_interval: int = 20
    """Save U_i every N sweeps. Initial and final states are always saved."""

    cfg_scale: float = 1.0
    """Classifier-free guidance scale. 1.0 = standard conditional (no CFG).
    >1 amplifies ligand conditioning, <1 dampens it."""

    active_site_only_cfg: bool = False
    """If True, classifier-free guidance is applied only at active-site
    positions (selected by active_site_mask). Non-active-site positions
    use the plain conditional distribution regardless of cfg_scale.
    Requires active_site_mask to be passed to sample()."""

    granular_active_site_coloring: bool = False
    """If True, each active-site token is assigned a unique colour during
    Gibbs graph colouring so that no two active-site residues end up in the
    same colour group (and are therefore never updated simultaneously).
    Requires active_site_mask to be passed to sample()."""

    @typecheck
    def sample(
        self,
        h: Float[Tensor, "batch n_token num_residue_types"],
        J: Float[
            Tensor, "batch n_token neighbours num_residue_types num_residue_types"
        ],
        edge_index: Int[Tensor, "batch n_token neighbours"],
        edge_mask: Bool[Tensor, "batch n_token neighbours"],
        token_mask: Bool[Tensor, "batch n_token"] | None = None,
        S_init: Float[Tensor, "batch n_token num_residue_types"] | None = None,
        generator: torch.Generator | None = None,
        penalty: Callable[[Tensor], Tensor] | None = None,
        h_uncond: Float[Tensor, "batch n_token num_residue_types"] | None = None,
        J_uncond: Float[
            Tensor,
            "batch n_token neighbours_uncond num_residue_types num_residue_types",
        ]
        | None = None,
        edge_index_uncond: Int[Tensor, "batch n_token neighbours_uncond"] | None = None,
        edge_mask_uncond: Bool[Tensor, "batch n_token neighbours_uncond"] | None = None,
        active_site_mask: Bool[Tensor, "batch n_token"] | None = None,
    ) -> PottsSamplingResult:
        """Sample sequences from the Potts energy landscape.

        Args:
            h: Potts fields [B, N, R].
            J: Potts couplings [B, N, K, R, R].
            edge_index: Neighbor indices [B, N, K].
            edge_mask: Valid edge mask [B, N, K].
            token_mask: Valid token mask [B, N]. Invalid positions are not updated.
            S_init: Initial one-hot sequence [B, N, R]. If None, initialised from field (h).
            generator: Generator for reproducibility.
            penalty: Optional penalty (callable) mapping S [B, N, R] to energy [B].
                Its gradient w.r.t. S is added to conditional energies during proposals.
            h_uncond: Unconditional Potts fields for CFG [B, N, R].
            J_uncond: Unconditional Potts couplings for CFG [B, N, K', R, R].
            edge_index_uncond: Unconditional neighbor indices [B, N, K'].
            edge_mask_uncond: Unconditional edge mask [B, N, K'].

        Returns:
            PottsSamplingResult with final sequence (one-hot) and energy.

        # TODO: changing the multinomial to gumbel max trick might be faster
        """
        B, N, R = h.shape

        if S_init is None:
            # initialise seq from field (h)
            logits = F.log_softmax(-h / self.temperature_init, dim=-1)
            flat_logits = logits.reshape(B * N, R)
            indices = torch.multinomial(
                flat_logits.exp(), 1, generator=generator
            ).reshape(B, N)
            S = F.one_hot(indices, R).float()
        else:
            S = S_init

        need_tracker = self.track_trajectory or self.track_u_i
        tracker = (
            TrajectoryTracker.create(
                self.n_sweeps,
                self.trajectory_interval,
                S,
                h,
                J,
                edge_index,
                edge_mask,
                u_i_interval=self.u_i_interval if self.track_u_i else None,
                penalty=penalty,
            )
            if need_tracker
            else None
        )

        # Pre-compute graph colouring for Gibbs
        color_masks = None
        if self.proposal == PottsProposal.GIBBS:
            color_masks = _color_graph(
                edge_index,
                edge_mask,
                active_site_mask=active_site_mask,
                granular_active_site_coloring=self.granular_active_site_coloring,
            )

        anneal_steps = int(self.annealing_fraction * self.n_sweeps)

        for t in range(self.n_sweeps):
            if anneal_steps > 0 and t < anneal_steps:
                frac = t / anneal_steps
                T = self.temperature_init + frac * (
                    self.temperature - self.temperature_init
                )
            else:
                T = self.temperature

            cfg_kwargs = dict(
                h_uncond=h_uncond,
                J_uncond=J_uncond,
                edge_index_uncond=edge_index_uncond,
                edge_mask_uncond=edge_mask_uncond,
                cfg_scale=self.cfg_scale,
                active_site_mask=active_site_mask,
                active_site_only_cfg=self.active_site_only_cfg,
            )

            if self.proposal == PottsProposal.GIBBS:
                result = _potts_proposal_gibbs(
                    S,
                    h,
                    J,
                    edge_index,
                    edge_mask,
                    token_mask,
                    T,
                    color_masks,
                    generator,
                    penalty,
                    **cfg_kwargs,
                )
                S = result.S_new
            elif self.proposal == PottsProposal.DLMC:
                result = _potts_proposal_dlmc(
                    S,
                    h,
                    J,
                    edge_index,
                    edge_mask,
                    token_mask,
                    T,
                    self.dt,
                    generator,
                    penalty,
                    **cfg_kwargs,
                )

                if self.rejection_step:
                    S = _mh_rejection_step(
                        S,
                        result,
                        h,
                        J,
                        edge_index,
                        edge_mask,
                        token_mask,
                        T,
                        self.dt,
                        penalty,
                        generator,
                        **cfg_kwargs,
                    )
                else:
                    S = result.S_new

            if tracker is not None:
                tracker.record(t + 1, S, h, J, edge_index, edge_mask)

        energy = compute_potts_energy(S.argmax(-1), h, J, edge_index, edge_mask)

        return PottsSamplingResult(
            sequence=S,
            energy=energy,
            sequence_trajectory=tracker.seq_traj if tracker else None,
            energy_trajectory=tracker.energy_traj if tracker else None,
            trajectory_sweep_indices=tracker.save_list if tracker else None,
            u_i_trajectory=tracker.u_i_traj if tracker else None,
            u_i_sweep_indices=tracker.u_i_save_list if tracker else None,
            penalty_grad_trajectories=tracker.penalty_grad_trajs if tracker else None,
            penalty_names=tracker.penalty_collection.names
            if tracker and tracker.penalty_collection
            else None,
        )
