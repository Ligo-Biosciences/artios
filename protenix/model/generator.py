from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from protenix.model.utils import beta_function
from protenix.utils.geometry import centre_random_augmentation
from protenix.utils.typing import Bool, Float, typecheck


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


@dataclass(frozen=True)
class BetaWithUniform(TrainingTimestepSampler):
    """Sample timesteps using a mixture of Beta(alpha,beta) and Uniform(0,1) distributions."""

    alpha: float
    beta: float
    uniform_weight: float
    eps: float = 1e-4

    @typecheck
    def sample(
        self,
        size: tuple[int, ...],
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "..."]:
        """Sample timesteps from the mixture distribution.

        Args:
            size: Shape of output tensor.
            device: Device to place tensor on.
            generator: Generator for reproducibility.

        Returns:
            Sampled timesteps tensor.
        """
        p_beta = 1.0 - self.uniform_weight

        # Component selector: 1 → take Beta, 0 → take Uniform
        take_beta = torch.rand(size, generator=generator, device=device) < p_beta

        # Beta(alpha, beta) distribution using torch.distributions
        beta_dist = torch.distributions.Beta(self.alpha, self.beta)

        # Sample from Beta(alpha, beta)
        beta_part = beta_dist.sample(size).to(device)

        # Sample from Uniform(0, 1)
        uniform_part = torch.rand(size, generator=generator, device=device)

        timesteps = torch.where(take_beta, beta_part, uniform_part)

        # Clamp timesteps to avoid numerical instability at t=0 or t=1
        return timesteps.clamp(min=self.eps, max=1.0 - self.eps)


@dataclass(frozen=True)
class ProteinaCoordSampler(TrainingTimestepSampler):
    """Sample timesteps from Beta(alpha, 1) + Uniform mixture with generator support.

    For Beta(alpha, 1), the CDF is F(x) = x^alpha, so inverse CDF is F^{-1}(u) = u^{1/alpha}.
    This allows deterministic sampling via a generator.
    """

    alpha: float = 1.9
    uniform_weight: float = 0.02
    eps: float = 1e-4

    @typecheck
    def sample(
        self,
        size: tuple[int, ...],
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "..."]:
        """Sample timesteps from the mixture distribution.

        Args:
            size: Shape of output tensor.
            device: Device to place tensor on.
            generator: Generator for reproducibility.

        Returns:
            Sampled timesteps tensor.
        """
        p_beta = 1.0 - self.uniform_weight
        take_beta = torch.rand(size, generator=generator, device=device) < p_beta

        # Sample from Beta(alpha, 1) using inverse CDF: F^{-1}(u) = u^{1/alpha}
        u = torch.rand(size, generator=generator, device=device)
        beta_part = u ** (1.0 / self.alpha)

        # Sample from Uniform(0, 1)
        uniform_part = torch.rand(size, generator=generator, device=device)
        timesteps = torch.where(take_beta, beta_part, uniform_part)

        # Clamp timesteps to avoid numerical instability at t=0 or t=1
        return timesteps.clamp(min=self.eps, max=1.0 - self.eps)


@dataclass(frozen=True)
class VariableLinQuadScheduler(InferenceTimestepScheduler):
    """Linear-quadratic timestep scheduler.

    Linear between points (0,0) and (m,n), then concave quadratic between (m,n) and (1,1).
    """

    m: float = 0.097
    n: float = 0.418

    @typecheck
    def get_schedule(
        self,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Float[Tensor, "n_steps_plus_one"]:
        """Generate timesteps using linear-quadratic schedule.

        Args:
            n_steps: Number of timesteps.
            device: Device to place tensor on.
            dtype: Data type of tensor.

        Returns:
            Timesteps tensor of shape [n_steps + 1].
        """
        steps = torch.arange(n_steps + 1, dtype=dtype, device=device) / n_steps
        linear_steps = steps[steps < self.m]
        quadratic_steps = steps[steps > self.m]

        # Create the timesteps on the linear section
        linear_t = linear_steps * (self.n / self.m)

        # Create the timesteps on the quadratic section
        a = (self.n - 1) / ((self.m - 1) ** 2)
        b = -2 * a
        c = a + 1
        quadratic_t = a * (quadratic_steps**2) + b * quadratic_steps + c

        # Concatenate all the timesteps
        t = torch.cat([linear_t, quadratic_t])
        return t


@dataclass(frozen=True)
class LaProteinaCoordScheduler(InferenceTimestepScheduler):
    """Timestep scheduler from the La-Proteina paper."""

    scale: float = 10.0

    @typecheck
    def get_schedule(
        self,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Float[Tensor, "n_steps_plus_one"]:
        """Generate timesteps using La-Proteina schedule.

        Args:
            n_steps: Number of timesteps.
            device: Device to place tensor on.
            dtype: Data type of tensor.

        Returns:
            Timesteps tensor of shape [n_steps + 1].
        """
        steps = torch.arange(n_steps + 1, dtype=dtype, device=device) / n_steps
        t = (1 - torch.pow(self.scale, -2 * steps)) / (1 - self.scale ** (-2))
        return t


# =============================================================================
# Stochasticity Schedules for SDE Integration
# =============================================================================
#
# These define g(t) in the SDE: dx = v*dt + g(t)*s*dt + sqrt(2*g(t)*gamma)*dW
# where v is velocity, s is score, and gamma is temperature.


class StochasticitySchedule(ABC):
    """Abstract base class for stochasticity schedules in flow matching SDEs.

    Defines g(t) in: dx = v*dt + g(t)*s*dt + sqrt(2*g(t)*gamma)*dW
    """

    @abstractmethod
    @typecheck
    def __call__(self, t: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        """Compute g(t) at timestep t.

        Args:
            t: Timesteps tensor of any shape.

        Returns:
            g(t) values with same shape as input.
        """
        ...


@dataclass(frozen=True)
class Deterministic(StochasticitySchedule):
    """No stochasticity: g(t) = 0 for all t."""

    @typecheck
    def __call__(self, t: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return torch.zeros_like(t)


@dataclass(frozen=True)
class Constant(StochasticitySchedule):
    """Constant stochasticity: g(t) = 0.5 * alpha^2.

    Integral: int_0^1 g(t) dt = 0.5 * alpha^2
    """

    alpha: float = 1.0

    @typecheck
    def __call__(self, t: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return torch.ones_like(t) * (0.5 * (self.alpha**2))


@dataclass(frozen=True)
class NonSingular(StochasticitySchedule):
    """Linear decay stochasticity with cutoff near t=1.

    g(t) = 4 * 0.5 * alpha^2 * (1 - t)  for t <= cutoff
    g(t) = 0                             for t > cutoff

    Integral (with cutoff=0.95): int_0^0.95 g(t) dt ≈ 0.5 * alpha^2
    """

    alpha: float = 1.0
    cutoff: float = 0.95

    @typecheck
    def __call__(self, t: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        g = 4 * 0.5 * (self.alpha**2) * (1 - t)
        return torch.where(t > self.cutoff, torch.zeros_like(t), g)


@dataclass(frozen=True)
class ZeroEnds(StochasticitySchedule):
    """Parabolic stochasticity that vanishes at t=0 and t=1.

    g(t) = 12 * 0.5 * alpha^2 * t * (1 - t)

    Integral: int_0^1 g(t) dt = 0.5 * alpha^2
    """

    alpha: float = 1.0

    @typecheck
    def __call__(self, t: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return 12 * 0.5 * (self.alpha**2) * t * (1 - t)


@dataclass(frozen=True)
class Proteina(StochasticitySchedule):
    """Proteina-style stochasticity with 1/t singularity.

    g(t) = 0.5 * alpha^2 / (t + eps)  for t <= cutoff
    g(t) = 0                           for t > cutoff
    """

    alpha: float = 1.0
    cutoff: float = 0.95
    eps: float = 0.01

    @typecheck
    def __call__(self, t: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        g = 0.5 * ((self.alpha**2) / (t + self.eps))
        return torch.where(t > self.cutoff, torch.zeros_like(t), g)


@dataclass(frozen=True)
class BetaSchedule(StochasticitySchedule):
    """Beta distribution-shaped stochasticity.

    g(t) = 2 * 0.5 * alpha^2 * Beta_pdf(t; mean, std)  for t <= cutoff
    g(t) = 0                                            for t > cutoff
    """

    alpha: float = 1.0
    beta_mean: float = 0.5
    beta_std: float = 0.224
    cutoff: float = 0.95

    @typecheck
    def __call__(self, t: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        g = (
            2
            * 0.5
            * (self.alpha**2)
            * beta_function(
                t, self.beta_mean, self.beta_std, device=t.device, dtype=t.dtype
            )
        )
        return torch.where(t > self.cutoff, torch.zeros_like(t), g)


@dataclass(frozen=True)
class LaProteina(StochasticitySchedule):
    """La-Proteina style stochasticity with tan singularity.

    g(t) = 0.5 * pi/2 * alpha^2 * tan(pi/2 * (1 - t) - eps)  for t <= cutoff
    g(t) = 0                                                   for t > cutoff
    """

    alpha: float = 1.0
    cutoff: float = 0.95
    eps: float = 0.01

    @typecheck
    def __call__(self, t: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        import math

        g = (
            0.5
            * math.pi
            / 2
            * (self.alpha**2)
            * torch.tan(math.pi / 2 * (1 - t) - self.eps)
        )
        return torch.where(t > self.cutoff, torch.zeros_like(t), g)


def stochasticity_from_config(
    stoch_type: str,
    alpha: float = 1.0,
    beta_mean: float = 0.5,
    beta_std: float = 0.224,
    cutoff: float = 0.95,
) -> StochasticitySchedule:
    """Create a StochasticitySchedule from config parameters.

    Args:
        stoch_type: One of "deterministic", "constant", "non_singular", "zero_ends",
            "proteina", "beta", "laproteina".
        alpha: Noise scale parameter.
        beta_mean: Mean for BetaSchedule.
        beta_std: Std for BetaSchedule.
        cutoff: Cutoff timestep for schedules with singularities.

    Returns:
        Configured StochasticitySchedule instance.

    Raises:
        ValueError: If stoch_type is not recognized.
    """
    schedules: dict[str, StochasticitySchedule] = {
        "deterministic": Deterministic(),
        "constant": Constant(alpha=alpha),
        "non_singular": NonSingular(alpha=alpha, cutoff=cutoff),
        "zero_ends": ZeroEnds(alpha=alpha),
        "proteina": Proteina(alpha=alpha, cutoff=cutoff),
        "beta": BetaSchedule(
            alpha=alpha, beta_mean=beta_mean, beta_std=beta_std, cutoff=cutoff
        ),
        "laproteina": LaProteina(alpha=alpha, cutoff=cutoff),
    }
    if stoch_type not in schedules:
        valid_types = ", ".join(schedules.keys())
        raise ValueError(
            f"Invalid stochasticity type: {stoch_type}. Must be one of: {valid_types}"
        )
    return schedules[stoch_type]


# =============================================================================
# Atomic Integration Primitives
# =============================================================================


@typecheck
def euler_maruyama_step(
    x: Float[Tensor, "... n_atom 3"],
    velocity: Float[Tensor, "... n_atom 3"],
    score: Float[Tensor, "... n_atom 3"],
    dt: float,
    g_t: float,
    gamma: float = 1.0,
    noise: Float[Tensor, "... n_atom 3"] | None = None,
) -> Float[Tensor, "... n_atom 3"]:
    """Perform a single Euler-Maruyama integration step for flow matching SDE.

    Integrates: dx = v*dt + g(t)*s*dt + sqrt(2*g(t)*γ)*dW

    Args:
        x: current coordinates.
        velocity: Predicted velocity at current state.
        score: Predicted score at current state.
        dt: Time step size (t_next - t_cur).
        g_t: Stochasticity schedule value at current timestep.
        gamma: Temperature parameter for stochastic term.
        noise: Optional pre-sampled noise. If None and g_t > 0, zeros are used
            (deterministic step).

    Returns:
        Updated coordinates after one integration step.
    """
    # Deterministic drift: v*dt + g(t)*s*dt
    drift = dt * (velocity + g_t * score)

    # Stochastic diffusion: sqrt(2*g(t)*γ)*dW
    if noise is not None and g_t > 0:
        diffusion = (2 * dt * g_t * gamma) ** 0.5 * noise
    else:
        diffusion = 0.0

    return x + drift + diffusion


@typecheck
def midpoint_step(
    x: Float[Tensor, "... n_atom 3"],
    velocity: Float[Tensor, "... n_atom 3"],
    score: Float[Tensor, "... n_atom 3"],
    velocity_mid: Float[Tensor, "... n_atom 3"],
    score_mid: Float[Tensor, "... n_atom 3"],
    dt: float,
    g_t: float,
    gamma: float = 1.0,
    noise: Float[Tensor, "... n_atom 3"] | None = None,
) -> Float[Tensor, "... n_atom 3"]:
    """Perform a midpoint (second-order) integration step.

    Uses velocity/score at both current and midpoint for higher accuracy.

    Args:
        x: Current coordinates.
        velocity: Predicted velocity at current state.
        score: Predicted score at current state.
        velocity_mid: Predicted velocity at midpoint.
        score_mid: Predicted score at midpoint.
        dt: Time step size.
        g_t: Stochasticity schedule value.
        gamma: Temperature parameter.
        noise: Optional pre-sampled noise.

    Returns:
        Updated coordinates after one midpoint step.
    """
    # TODO: this is not the midpoint step, fix!
    # Average the updates from current and midpoint
    avg_velocity = 0.5 * (velocity + velocity_mid)
    avg_score = 0.5 * (score + score_mid)

    drift = dt * (avg_velocity + g_t * avg_score)

    if noise is not None and g_t > 0:
        diffusion = (2 * dt * g_t * gamma) ** 0.5 * noise
    else:
        diffusion = 0.0

    return x + drift + diffusion


@runtime_checkable
class VelocityScorePredictor(Protocol):
    """Protocol for models that predict velocity and score from noisy coordinates.

    Any callable or class implementing this signature can be used with
    FlowMatchingSampler. The protocol uses structural typing, so no inheritance
    is required - just implement the __call__ method with matching signature.

    Example:
        >>> # Create predictor from a denoising network
        >>> def my_predictor(x_noisy, timestep):
        ...     pred = model(x_noisy, timestep)
        ...     velocity = (pred - x_noisy) / (1 - timestep[..., None, None])
        ...     score = timestep[..., None, None] * velocity - x_noisy
        ...     return velocity, score
        >>>
        >>> sampler = FlowMatchingSampler(timestep_scheduler=LaProteinaCoordScheduler())
        >>> result = sampler.sample(my_predictor, x_init, n_steps=50)
    """

    @typecheck
    def __call__(
        self,
        x_noisy: Float[Tensor, "*batch n_atom 3"],
        timestep: Float[Tensor, "*batch"],
    ) -> tuple[Float[Tensor, "*batch n_atom 3"], Float[Tensor, "*batch n_atom 3"]]:
        """Predict velocity and score at given noisy coordinates and timestep.

        Args:
            x_noisy: Noisy coordinates.
            timestep: Current diffusion timestep (0 = noise, 1 = data).

        Returns:
            Tuple of (velocity, score) tensors with same shape as x_noisy.
        """
        ...


@dataclass(frozen=True)
class FlowMatchingSampler:
    """Configurable sampler for flow matching SDEs."""

    timestep_scheduler: InferenceTimestepScheduler
    """Scheduler that generates the timestep sequence for sampling."""

    stochasticity: StochasticitySchedule = Deterministic()
    """Schedule defining g(t) in the SDE. Defaults to deterministic (ODE) sampling."""

    gamma: float = 1.0
    """Temperature parameter for the stochastic term."""

    second_order: bool = False
    """Use midpoint method for higher-order accuracy."""

    centre_coords: bool = True
    """Apply centre_random_augmentation at each step."""

    @typecheck
    def sample(
        self,
        predictor: VelocityScorePredictor,
        x_init: Float[Tensor, "... n_atom 3"],
        n_steps: int,
        mask: Bool[Tensor, "... n_atom"] | None = None,
    ) -> Float[Tensor, "... n_atom 3"]:
        """Sample from the flow matching SDE.

        Args:
            predictor: Callable that returns (velocity, score) given (x_noisy, timestep).
            x_init: Initial noisy coordinates (typically Gaussian noise scaled by sigma).
            n_steps: Number of integration steps.
            mask: Optional atom mask for coordinate centering.

        Returns:
            Final denoised coordinates.
        """
        device = x_init.device
        dtype = x_init.dtype

        # Get timestep schedule
        timesteps = self.timestep_scheduler.get_schedule(n_steps, device, dtype)

        # Precompute g(t) for all timesteps
        g_values = self.stochasticity(timesteps)
        is_deterministic = isinstance(self.stochasticity, Deterministic)

        # Initialize
        x = x_init

        for i, (t_cur, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            dt = (t_next - t_cur).item()
            g_t = g_values[i].item()

            # Generate noise for stochastic sampling
            noise = None if is_deterministic else torch.randn_like(x)

            # Centre coordinates if requested
            if self.centre_coords:
                x = (
                    centre_random_augmentation(x_input_coords=x, n_sample=1, mask=mask)
                    .squeeze(dim=-3)
                    .to(dtype)
                )

            # Expand timestep to match batch dimensions
            # TODO: I don't like the .item() calls
            t_expanded = x.new_full(x.shape[:-2], t_cur.item())

            # Get velocity and score predictions
            velocity, score = predictor(x, t_expanded)

            if not self.second_order:
                x = euler_maruyama_step(
                    x=x,
                    velocity=velocity,
                    score=score,
                    dt=dt,
                    g_t=g_t,
                    gamma=self.gamma,
                    noise=noise,
                )
            else:
                # TODO: refine this (Oli)
                # Midpoint method: first get midpoint position
                x_mid = euler_maruyama_step(
                    x=x,
                    velocity=velocity,
                    score=score,
                    dt=dt,
                    g_t=g_t,
                    gamma=0.0,  # No noise for midpoint estimate
                    noise=None,
                )
                velocity_mid, score_mid = predictor(x_mid, t_expanded)
                x = midpoint_step(
                    x=x,
                    velocity=velocity,
                    score=score,
                    velocity_mid=velocity_mid,
                    score_mid=score_mid,
                    dt=dt,
                    g_t=g_t,
                    gamma=self.gamma,
                    noise=noise,
                )

        return x
