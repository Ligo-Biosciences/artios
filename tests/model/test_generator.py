import pytest
import torch

from artios.model.generator import (
    EDMInferenceNoiseScheduler,
    EDMSampler,
    EDMTrainingNoiseSampler,
)


class FixedNoiseScheduler:
    def __init__(self, schedule: list[float]):
        self.schedule = schedule

    def get_schedule(
        self,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        assert n_steps == len(self.schedule) - 1
        return torch.tensor(self.schedule, device=device, dtype=dtype)


class TestEDMGenerator:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_training_noise_sampler_matches_lognormal_formula(self, device):
        sampler = EDMTrainingNoiseSampler()
        size = (2, 5)
        seed = 123

        generator = torch.Generator(device=device).manual_seed(seed)
        noise_level = sampler.sample(size=size, device=device, generator=generator)

        expected_generator = torch.Generator(device=device).manual_seed(seed)
        rnd_normal = torch.randn(size=size, device=device, generator=expected_generator)
        expected = torch.exp(rnd_normal * 1.5 - 1.2) * 16.0

        torch.testing.assert_close(noise_level, expected)
        assert (noise_level > 0).all()

    def test_inference_noise_scheduler_matches_af3_formula(self, device):
        scheduler = EDMInferenceNoiseScheduler()
        n_steps = 8

        schedule = scheduler.get_schedule(n_steps=n_steps, device=device)

        step_indices = torch.arange(n_steps + 1, device=device)
        expected = (
            16.0
            * (
                160.0 ** (1 / 7.0)
                + step_indices / n_steps * (4e-4 ** (1 / 7.0) - 160.0 ** (1 / 7.0))
            )
            ** 7.0
        )
        expected[-1] = 0

        torch.testing.assert_close(schedule, expected)
        assert torch.all(torch.diff(schedule) < 0)
        assert schedule[-1] == 0

    def test_edm_params_are_not_configurable(self):
        with pytest.raises(TypeError):
            EDMTrainingNoiseSampler(p_mean=-1.2)

        with pytest.raises(TypeError):
            EDMInferenceNoiseScheduler(s_max=160.0)

    def test_edm_sampler_params_are_not_configurable(self):
        with pytest.raises(TypeError):
            EDMSampler(noise_scheduler=FixedNoiseScheduler([1.0, 0.0]), gamma0=0.5)

        with pytest.raises(TypeError):
            EDMSampler(
                noise_scheduler=FixedNoiseScheduler([1.0, 0.0]),
                step_scale_eta=2.0,
            )
