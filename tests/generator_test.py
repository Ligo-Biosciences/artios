import pytest
import torch

from protenix.model.generator import (
    BetaSchedule,
    BetaWithUniform,
    Constant,
    Deterministic,
    FlowMatchingSampler,
    LaProteina,
    LaProteinaCoordScheduler,
    NonSingular,
    Proteina,
    VariableLinQuadScheduler,
    ZeroEnds,
    euler_maruyama_step,
    midpoint_step,
    sample_noising,
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

    def test_sample_noising_basic(self, device):
        """Test basic functionality of sample_noising function."""
        batch_shape = (1,)
        n_tokens = 100
        n_features = 3

        input_coords = torch.randn(*batch_shape, n_tokens, n_features, device=device)
        mask = torch.ones(*batch_shape, n_tokens, device=device, dtype=torch.bool)

        timestep_sampler = BetaWithUniform(alpha=2.2, beta=1.0, uniform_weight=0.02)

        num_sample = 3
        scale = 12.0
        generator = torch.Generator(device=device).manual_seed(42)

        input_noisy, timestep, input_augment = sample_noising(
            input=input_coords,
            mask=mask,
            timestep_sampler=timestep_sampler,
            num_sample=num_sample,
            scale=scale,
            generator=generator,
        )

        expected_shape = (*batch_shape, num_sample, n_tokens, n_features)
        expected_timestep_shape = (*batch_shape, num_sample)

        assert input_noisy.shape == expected_shape
        assert input_augment.shape == expected_shape
        assert timestep.shape == expected_timestep_shape

        assert not torch.isnan(input_noisy).any()
        assert not torch.isnan(input_augment).any()
        assert not torch.isnan(timestep).any()
        assert ((timestep >= 0) & (timestep <= 1)).all()
        assert not torch.allclose(input_noisy, input_augment)

    def test_sample_noising_different_scales(self, device):
        """Test sample_noising with different noise scales."""
        batch_shape = (1,)
        n_tokens = 30
        n_features = 3

        input_coords = torch.randn(*batch_shape, n_tokens, n_features, device=device)
        mask = torch.ones(*batch_shape, n_tokens, device=device, dtype=torch.bool)

        timestep_sampler = BetaWithUniform(alpha=2.2, beta=1.0, uniform_weight=0.02)
        num_sample = 1

        scales = [1.0, 5.0, 20.0]
        results = []

        for scale in scales:
            generator = torch.Generator(device=device).manual_seed(42)
            result = sample_noising(
                input=input_coords,
                mask=mask,
                timestep_sampler=timestep_sampler,
                num_sample=num_sample,
                scale=scale,
                generator=generator,
            )
            results.append(result)

        for i in range(len(scales) - 1):
            assert results[i][0].shape == results[i + 1][0].shape
            assert not torch.isnan(results[i][0]).any()
            assert not torch.isnan(results[i + 1][0]).any()

    def test_sample_noising_edge_cases(self, device):
        """Test sample_noising with edge cases."""
        batch_shape = (1,)
        n_tokens = 10
        n_features = 3

        input_coords = torch.randn(*batch_shape, n_tokens, n_features, device=device)
        mask = torch.ones(*batch_shape, n_tokens, device=device, dtype=torch.bool)

        timestep_sampler = BetaWithUniform(alpha=2.2, beta=1.0, uniform_weight=0.02)
        generator = torch.Generator(device=device).manual_seed(0)

        # Test with num_sample = 1
        result = sample_noising(
            input=input_coords,
            mask=mask,
            timestep_sampler=timestep_sampler,
            num_sample=1,
            scale=1.0,
            generator=generator,
        )

        assert result[0].shape == (*batch_shape, 1, n_tokens, n_features)
        assert result[1].shape == (*batch_shape, 1)
        assert result[2].shape == (*batch_shape, 1, n_tokens, n_features)

        # Test with partial mask
        partial_mask = torch.zeros(
            *batch_shape, n_tokens, device=device, dtype=torch.bool
        )
        partial_mask[..., : n_tokens // 2] = True

        generator = torch.Generator(device=device).manual_seed(0)
        result_partial = sample_noising(
            input=input_coords,
            mask=partial_mask,
            timestep_sampler=timestep_sampler,
            num_sample=1,
            scale=1.0,
            generator=generator,
        )

        assert result_partial[0].shape == (*batch_shape, 1, n_tokens, n_features)
        assert not torch.isnan(result_partial[0]).any()

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


if __name__ == "__main__":
    pytest.main([__file__])
