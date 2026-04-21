import pytest
import torch

from artios.model.generator import (
    Constant,
    Deterministic,
    LaProteinaCoordScheduler,
    NonSingular,
    Proteina,
    VariableLinQuadScheduler,
    ZeroEnds,
    euler_maruyama_step,
    midpoint_step,
)


class TestGeneratorBackbone:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_new_schedulers(self, device):
        """Test scheduler classes."""
        n_steps = 50

        # Test VariableLinQuadScheduler
        vl_scheduler = VariableLinQuadScheduler(m=0.097, n=0.418)
        vl_timesteps = vl_scheduler.get_schedule(n_steps=n_steps, device=device)
        assert vl_timesteps.shape == (n_steps + 1,)
        assert vl_timesteps[0] == 0
        assert torch.isclose(vl_timesteps[-1], torch.tensor(1.0, device=device))
        assert torch.all(torch.diff(vl_timesteps) > 0)

        # Test LaProteinaCoordScheduler
        lp_coord_scheduler = LaProteinaCoordScheduler(scale=2.0)
        lp_coord_timesteps = lp_coord_scheduler.get_schedule(
            n_steps=n_steps, device=device
        )
        assert lp_coord_timesteps.shape == (n_steps + 1,)
        assert lp_coord_timesteps[0] == 0
        assert torch.isclose(lp_coord_timesteps[-1], torch.tensor(1.0, device=device))
        assert torch.all(torch.diff(lp_coord_timesteps) > 0)


class TestStochasticitySchedules:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_deterministic(self, device):
        """Test Deterministic schedule returns zeros."""
        schedule = Deterministic()
        t = torch.linspace(0, 1, 10, device=device)
        g = schedule(t)
        assert torch.all(g == 0)

    def test_constant(self, device):
        """Test Constant schedule."""
        alpha = 2.0
        schedule = Constant(alpha=alpha)
        t = torch.linspace(0, 1, 10, device=device)
        g = schedule(t)
        expected = 0.5 * alpha**2
        assert torch.allclose(g, torch.full_like(g, expected))

    def test_non_singular(self, device):
        """Test NonSingular schedule with cutoff."""
        schedule = NonSingular(alpha=1.0, cutoff=0.95)
        t = torch.tensor([0.0, 0.5, 0.94, 0.96, 1.0], device=device)
        g = schedule(t)

        # Values at t=0 should be highest
        assert g[0] > g[1] > g[2]
        # Values above cutoff should be zero
        assert g[3] == 0
        assert g[4] == 0

    def test_zero_ends(self, device):
        """Test ZeroEnds schedule vanishes at boundaries."""
        schedule = ZeroEnds(alpha=1.0)
        t = torch.tensor([0.0, 0.5, 1.0], device=device)
        g = schedule(t)

        assert g[0] == 0  # g(0) = 0
        assert g[2] == 0  # g(1) = 0
        assert g[1] > 0  # g(0.5) > 0

    def test_proteina(self, device):
        """Test Proteina schedule with 1/t singularity."""
        schedule = Proteina(alpha=1.0, cutoff=0.95, eps=0.01)
        t = torch.tensor([0.01, 0.5, 0.94, 0.96], device=device)
        g = schedule(t)

        # Should decay as t increases
        assert g[0] > g[1] > g[2]
        # Above cutoff should be zero
        assert g[3] == 0


class TestIntegrationPrimitives:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_euler_maruyama_step_deterministic(self, device):
        """Test Euler-Maruyama step with no noise."""
        batch_shape = (2,)
        n_atom = 10

        x = torch.randn(*batch_shape, n_atom, 3, device=device)
        velocity = torch.randn_like(x)
        score = torch.randn_like(x)
        dt = 0.1
        g_t = 0.5

        x_next = euler_maruyama_step(
            x=x, velocity=velocity, score=score, dt=dt, g_t=g_t, gamma=1.0, noise=None
        )

        # Without noise, should just be drift
        expected = x + dt * (velocity + g_t * score)
        assert torch.allclose(x_next, expected)

    def test_euler_maruyama_step_stochastic(self, device):
        """Test Euler-Maruyama step with noise."""
        batch_shape = (2,)
        n_atom = 10

        x = torch.randn(*batch_shape, n_atom, 3, device=device)
        velocity = torch.randn_like(x)
        score = torch.randn_like(x)
        noise = torch.randn_like(x)
        dt = 0.1
        g_t = 0.5
        gamma = 1.0

        x_next = euler_maruyama_step(
            x=x,
            velocity=velocity,
            score=score,
            dt=dt,
            g_t=g_t,
            gamma=gamma,
            noise=noise,
        )

        drift = dt * (velocity + g_t * score)
        diffusion = (2 * dt * g_t * gamma) ** 0.5 * noise
        expected = x + drift + diffusion

        assert torch.allclose(x_next, expected)

    def test_midpoint_step(self, device):
        """Test midpoint integration step."""
        batch_shape = (2,)
        n_atom = 10

        x = torch.randn(*batch_shape, n_atom, 3, device=device)
        velocity = torch.randn_like(x)
        score = torch.randn_like(x)
        velocity_mid = torch.randn_like(x)
        score_mid = torch.randn_like(x)
        dt = 0.1
        g_t = 0.5

        x_next = midpoint_step(
            x=x,
            velocity=velocity,
            score=score,
            velocity_mid=velocity_mid,
            score_mid=score_mid,
            dt=dt,
            g_t=g_t,
            gamma=1.0,
            noise=None,
        )

        avg_velocity = 0.5 * (velocity + velocity_mid)
        avg_score = 0.5 * (score + score_mid)
        expected = x + dt * (avg_velocity + g_t * avg_score)

        assert torch.allclose(x_next, expected)


if __name__ == "__main__":
    pytest.main([__file__])
