import pytest
import torch

from artios.model.generator import (
    BetaSchedule,
    BetaWithUniform,
    CFGPredictor,
    Constant,
    ConstantCFG,
    Deterministic,
    FlowMatchingSampler,
    LaProteina,
    LaProteinaCoordScheduler,
    LinearCFG,
    NoCFG,
    NonSingular,
    Proteina,
    VariableLinQuadScheduler,
    ZeroEnds,
    cfg_schedule_from_config,
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

    def test_sampler_deterministic(self, device):
        """Test FlowMatchingSampler with deterministic sampling."""
        n_atom = 20
        n_steps = 10

        # Simple predictor that returns constant velocity/score
        def mock_predictor(x_noisy, timestep):
            velocity = -x_noisy  # Push towards origin
            score = torch.zeros_like(x_noisy)
            return velocity, score

        scheduler = LaProteinaCoordScheduler(scale=2.0)
        sampler = FlowMatchingSampler(
            timestep_scheduler=scheduler,
            stochasticity=Deterministic(),
            gamma=1.0,
            second_order=False,
            centre_coords=False,  # Disable for predictable testing
        )

        x_init = torch.randn(1, n_atom, 3, device=device)
        result = sampler.sample(
            predictor=mock_predictor, x_init=x_init, n_steps=n_steps
        )

        assert result.shape == x_init.shape
        assert not torch.isnan(result).any()

    def test_sampler_stochastic(self, device):
        """Test FlowMatchingSampler with stochastic sampling."""
        n_atom = 20
        n_steps = 10

        def mock_predictor(x_noisy, timestep):
            velocity = -x_noisy
            score = torch.zeros_like(x_noisy)
            return velocity, score

        scheduler = LaProteinaCoordScheduler(scale=2.0)
        sampler = FlowMatchingSampler(
            timestep_scheduler=scheduler,
            stochasticity=NonSingular(alpha=0.5),
            gamma=1.0,
            second_order=False,
            centre_coords=False,
        )

        x_init = torch.randn(1, n_atom, 3, device=device)
        result = sampler.sample(
            predictor=mock_predictor, x_init=x_init, n_steps=n_steps
        )

        assert result.shape == x_init.shape
        assert not torch.isnan(result).any()

    def test_sampler_second_order(self, device):
        """Test FlowMatchingSampler with second-order integration."""
        n_atom = 20
        n_steps = 10

        def mock_predictor(x_noisy, timestep):
            velocity = -x_noisy
            score = torch.zeros_like(x_noisy)
            return velocity, score

        scheduler = LaProteinaCoordScheduler(scale=2.0)
        sampler = FlowMatchingSampler(
            timestep_scheduler=scheduler,
            stochasticity=Deterministic(),
            gamma=1.0,
            second_order=True,
            centre_coords=False,
        )

        x_init = torch.randn(1, n_atom, 3, device=device)
        result = sampler.sample(
            predictor=mock_predictor, x_init=x_init, n_steps=n_steps
        )

        assert result.shape == x_init.shape
        assert not torch.isnan(result).any()

    def test_sampler_all_stochasticity_types(self, device):
        """Test FlowMatchingSampler with all stochasticity types."""
        n_atom = 20
        n_steps = 5

        def mock_predictor(x_noisy, timestep):
            return -x_noisy, torch.zeros_like(x_noisy)

        scheduler = LaProteinaCoordScheduler(scale=2.0)
        stochasticity_types = [
            Deterministic(),
            Constant(alpha=0.5),
            NonSingular(alpha=0.5),
            ZeroEnds(alpha=0.5),
            Proteina(alpha=0.5),
            BetaSchedule(alpha=0.5),
            LaProteina(alpha=0.5),
        ]

        for stoch in stochasticity_types:
            sampler = FlowMatchingSampler(
                timestep_scheduler=scheduler,
                stochasticity=stoch,
                gamma=1.0,
                centre_coords=False,
            )

            x_init = torch.randn(1, n_atom, 3, device=device)
            result = sampler.sample(
                predictor=mock_predictor, x_init=x_init, n_steps=n_steps
            )

            assert result.shape == x_init.shape, f"Failed for {type(stoch).__name__}"
            assert not torch.isnan(result).any(), f"NaN for {type(stoch).__name__}"


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

    def test_cfg_schedule_from_config(self, device):
        """Test cfg_schedule_from_config factory function."""
        # Test none
        schedule = cfg_schedule_from_config(cfg_type="none")
        assert isinstance(schedule, NoCFG)
        t = torch.tensor([0.5], device=device)
        assert schedule(t).item() == 1.0

        # Test constant
        schedule = cfg_schedule_from_config(cfg_type="constant", weight=1.5)
        assert isinstance(schedule, ConstantCFG)
        t = torch.tensor([0.5], device=device)
        assert schedule(t).item() == 1.5

        # Test linear
        schedule = cfg_schedule_from_config(
            cfg_type="linear",
            lower_guidance_limit_weight=2.0,
            upper_guidance_limit_weight=1.0,
            lower_guidance_limit=0.3,
            upper_guidance_limit=0.7,
        )
        assert isinstance(schedule, LinearCFG)

        # Test invalid type
        with pytest.raises(ValueError):
            cfg_schedule_from_config(cfg_type="invalid")


class TestCFGPredictor:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_cfg_scale_1_equals_conditional(self, device):
        """Test that CFG with scale=1.0 equals just the conditional predictor.

        When w=1: v_cfg = v_uncond + 1*(v_cond - v_uncond) = v_cond

        This test verifies both:
        1. Direct predictor outputs match
        2. Full sampling loop produces identical results
        """
        n_atom = 20
        batch_shape = (2,)

        # Create mock predictors with different outputs
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
            guidance_schedule=ConstantCFG(weight=1.0),
        )

        # Test 1: Direct predictor outputs match
        x = torch.randn(*batch_shape, n_atom, 3, device=device)
        t = torch.tensor([0.5, 0.7], device=device)

        v_cfg, s_cfg = cfg_predictor(x, t)
        v_cond, s_cond = cond_predictor(x, t)

        assert torch.allclose(v_cfg, v_cond)
        assert torch.allclose(s_cfg, s_cond)

        # Test 2: Full sampling loop produces identical results
        n_steps = 10
        scheduler = LaProteinaCoordScheduler(scale=2.0)
        sampler = FlowMatchingSampler(
            timestep_scheduler=scheduler,
            stochasticity=Deterministic(),
            centre_coords=False,  # Disable for deterministic comparison
        )

        # Use same initial noise for both samplers
        torch.manual_seed(42)
        x_init = torch.randn(1, n_atom, 3, device=device)

        # Sample with CFG predictor (scale=1.0)
        torch.manual_seed(42)
        x_init_cfg = x_init.clone()
        result_cfg = sampler.sample(
            predictor=cfg_predictor, x_init=x_init_cfg, n_steps=n_steps
        )

        # Sample with just the conditional predictor
        torch.manual_seed(42)
        x_init_cond = x_init.clone()
        result_cond = sampler.sample(
            predictor=cond_predictor, x_init=x_init_cond, n_steps=n_steps
        )

        assert torch.allclose(result_cfg, result_cond, atol=1e-6), (
            f"CFG with scale=1.0 should produce identical results to conditional predictor. "
            f"Max diff: {(result_cfg - result_cond).abs().max()}"
        )

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

    def test_cfg_in_sampler(self, device):
        """Test CFGPredictor integrates correctly with FlowMatchingSampler."""
        n_atom = 20
        n_steps = 5

        def cond_predictor(x_noisy, timestep):
            velocity = -x_noisy  # Push towards origin
            score = torch.zeros_like(x_noisy)
            return velocity, score

        def uncond_predictor(x_noisy, timestep):
            velocity = -x_noisy * 0.5  # Weaker push
            score = torch.zeros_like(x_noisy)
            return velocity, score

        cfg_predictor = CFGPredictor(
            cond_predictor=cond_predictor,
            uncond_predictor=uncond_predictor,
            guidance_schedule=ConstantCFG(weight=1.5),
        )

        scheduler = LaProteinaCoordScheduler(scale=2.0)
        sampler = FlowMatchingSampler(
            timestep_scheduler=scheduler,
            stochasticity=Deterministic(),
            centre_coords=False,
        )

        x_init = torch.randn(1, n_atom, 3, device=device)
        result = sampler.sample(predictor=cfg_predictor, x_init=x_init, n_steps=n_steps)

        assert result.shape == x_init.shape
        assert not torch.isnan(result).any()


if __name__ == "__main__":
    pytest.main([__file__])
