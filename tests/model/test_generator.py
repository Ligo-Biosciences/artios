import pytest
import torch

from artios.model.generator import (
    BetaSchedule,
    BetaWithUniform,
    CFGPredictor,
    Constant,
    ConstantCFG,
    Deterministic,
    LaProteina,
    LaProteinaCoordScheduler,
    LinearCFG,
    NoCFG,
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

    def test_training_timestep_sampler(self, device):
        """Test BetaWithUniform timestep sampler."""
        sampler = BetaWithUniform(alpha=2.2, beta=1.0, uniform_weight=0.02)

        batch_size = (2, 3)
        timesteps = sampler.sample(size=batch_size, device=device)

        assert timesteps.shape == batch_size
        assert ((timesteps >= 0) & (timesteps <= 1)).all()
        assert not torch.isnan(timesteps).any()

    def test_inference_timestep_scheduler(self, device):
        """Test LaProteinaCoordScheduler."""
        scheduler = LaProteinaCoordScheduler(scale=2.0)

        n_steps = 50
        timesteps = scheduler.get_schedule(n_steps=n_steps, device=device)

        assert timesteps.shape == (n_steps + 1,)
        assert timesteps[0] == 0
        assert torch.isclose(timesteps[-1], torch.tensor(1.0, device=device))
        assert torch.all(torch.diff(timesteps) > 0)  # Monotonically increasing
        assert not torch.isnan(timesteps).any()

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

    def test_beta_schedule(self, device):
        """Test BetaSchedule."""
        schedule = BetaSchedule(alpha=1.0, beta_mean=0.5, beta_std=0.224, cutoff=0.95)
        t = torch.linspace(0.01, 0.94, 10, device=device)
        g = schedule(t)

        assert not torch.isnan(g).any()
        assert torch.all(g >= 0)

    def test_laproteina(self, device):
        """Test LaProteina schedule with tan singularity."""
        schedule = LaProteina(alpha=1.0, cutoff=0.95, eps=0.01)
        t = torch.tensor([0.01, 0.5, 0.94, 0.96], device=device)
        g = schedule(t)

        assert not torch.isnan(g[:3]).any()
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

    def test_euler_maruyama_step_zero_g(self, device):
        """Test Euler-Maruyama with g_t=0 ignores noise."""
        batch_shape = (2,)
        n_atom = 10

        x = torch.randn(*batch_shape, n_atom, 3, device=device)
        velocity = torch.randn_like(x)
        score = torch.randn_like(x)
        noise = torch.randn_like(x) * 1000  # Large noise

        x_next = euler_maruyama_step(
            x=x, velocity=velocity, score=score, dt=0.1, g_t=0.0, gamma=1.0, noise=noise
        )

        # With g_t=0, noise should be ignored
        expected = x + 0.1 * velocity
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


class TestFlowMatchingSampler:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCFGSchedules:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_constant_cfg(self, device):
        """Test ConstantCFG returns constant scale."""
        scale = 1.5
        schedule = ConstantCFG(weight=scale)
        t = torch.linspace(0, 1, 10, device=device)
        w = schedule(t)

        assert torch.allclose(w, torch.full_like(w, scale))

    def test_no_cfg(self, device):
        """Test NoCFG returns 1.0 for all timesteps."""
        schedule = NoCFG()
        t = torch.linspace(0, 1, 10, device=device)
        w = schedule(t)

        assert torch.allclose(w, torch.ones_like(w))

    def test_linear_cfg(self, device):
        """Test LinearCFG applies guidance only within window."""
        schedule = LinearCFG(
            lower_guidance_limit_weight=2.0,
            upper_guidance_limit_weight=1.0,
            lower_guidance_limit=0.3,
            upper_guidance_limit=0.7,
        )
        t = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0], device=device)
        w = schedule(t)

        # Outside window: w = 1.0
        assert torch.isclose(w[0], torch.tensor(1.0, device=device))
        assert torch.isclose(w[4], torch.tensor(1.0, device=device))
        # At lower_guidance_limit: w = lower_guidance_limit_weight
        assert torch.isclose(w[1], torch.tensor(2.0, device=device))
        # At upper_guidance_limit: w = upper_guidance_limit_weight
        assert torch.isclose(w[3], torch.tensor(1.0, device=device))
        # In middle: linear interpolation
        assert torch.isclose(w[2], torch.tensor(1.5, device=device))


class TestCFGPredictor:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_cfg_scale_0_equals_unconditional(self, device):
        """Test that CFG with scale=0.0 equals just the unconditional predictor.

        When w=0: v_cfg = v_uncond + 0*(v_cond - v_uncond) = v_uncond
        """
        n_atom = 20
        batch_shape = (2,)

        def cond_predictor(x_noisy, timestep):
            velocity = torch.ones_like(x_noisy) * 2.0
            score = torch.ones_like(x_noisy) * 0.5
            return velocity, score

        def uncond_predictor(x_noisy, timestep):
            velocity = torch.ones_like(x_noisy) * -1.0
            score = torch.ones_like(x_noisy) * -0.5
            return velocity, score

        cfg_predictor = CFGPredictor(
            cond_predictor=cond_predictor,
            uncond_predictor=uncond_predictor,
            guidance_schedule=ConstantCFG(weight=0.0),
        )

        x = torch.randn(*batch_shape, n_atom, 3, device=device)
        t = torch.tensor([0.5, 0.7], device=device)

        v_cfg, s_cfg = cfg_predictor(x, t)
        v_uncond, s_uncond = uncond_predictor(x, t)

        assert torch.allclose(v_cfg, v_uncond)
        assert torch.allclose(s_cfg, s_uncond)

    def test_cfg_weighted_combination(self, device):
        """Test CFG produces correct weighted combination."""
        n_atom = 20
        batch_shape = (2,)

        def cond_predictor(x_noisy, timestep):
            velocity = torch.ones_like(x_noisy) * 2.0
            score = torch.ones_like(x_noisy) * 1.0
            return velocity, score

        def uncond_predictor(x_noisy, timestep):
            velocity = torch.zeros_like(x_noisy)
            score = torch.zeros_like(x_noisy)
            return velocity, score

        scale = 1.5
        cfg_predictor = CFGPredictor(
            cond_predictor=cond_predictor,
            uncond_predictor=uncond_predictor,
            guidance_schedule=ConstantCFG(weight=scale),
        )

        x = torch.randn(*batch_shape, n_atom, 3, device=device)
        t = torch.tensor([0.5, 0.7], device=device)

        v_cfg, s_cfg = cfg_predictor(x, t)

        # v_cfg = v_uncond + w * (v_cond - v_uncond) = 0 + 1.5 * (2 - 0) = 3.0
        expected_v = torch.ones_like(x) * 3.0
        # s_cfg = s_uncond + w * (s_cond - s_uncond) = 0 + 1.5 * (1 - 0) = 1.5
        expected_s = torch.ones_like(x) * 1.5

        assert torch.allclose(v_cfg, expected_v)
        assert torch.allclose(s_cfg, expected_s)


if __name__ == "__main__":
    pytest.main([__file__])
