from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn as nn
from ml_collections import ConfigDict

import artios.model.artios_backbone as backbone_module
from artios.model.artios_backbone import (
    ArtiosBackbone,
    ArtiosBackboneConfig,
    ArtiosBackboneRuntimeConfig,
)
from artios.model.generator import SIGMA_DATA
from artios.model.loss import edm_noise_level_scale
from artios.model.modules.diffusion_backbone import (
    DiffusionConditioning,
    DiffusionConditioningConfig,
    DiffusionModule,
    DiffusionModuleConfig,
    DiffusionRuntimeConfig,
)
from artios.model.modules.transformer import DiffusionTransformerConfig
from artios.model.output import DiffusionOutput
from tests._helpers.context import create_test_context


class ConstantNoiseSampler:
    def __init__(self, value: float):
        self.value = value

    def sample(self, size, device, generator=None):
        return torch.full(size, self.value, device=device)


class RecordingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_noisy = None
        self.noise_level = None

    def forward(
        self,
        r_noisy,
        noise_level,
        context,
        cache=None,
        inplace_safe=False,
        prev_pred=None,
    ):
        self.r_noisy = r_noisy.detach().clone()
        self.noise_level = noise_level.detach().clone()
        return r_noisy


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def context(device):
    return create_test_context(batch_size=1, n_tokens=8, n_atoms=16, device=device)


@pytest.fixture
def backbone_configs():
    return ConfigDict(
        {
            "checkpoint_grads": False,
            "diffusion_batch_size": 2,
            "self_conditioning_prob": 0.0,
            "sample_diffusion": {
                "N_step": 2,
                "N_sample": 1,
                "gamma0": 0.8,
                "gamma_min": 1.0,
                "noise_scale_lambda": 1.003,
                "step_scale_eta": 1.5,
                "cfg_type": "no-guidance",
                "cfg_weight": 1.0,
                "lower_guidance_limit_weight": 1.0,
                "upper_guidance_limit_weight": 1.0,
                "lower_guidance_limit": 0.0,
                "upper_guidance_limit": 1.0,
                "centre_and_augment_coords_during_sampling": False,
                "track_trajectory": False,
                "self_conditioning": True,
            },
            "loss": {
                "backbone": {
                    "mse": {
                        "weight_protein": 0.0,
                        "weight_dna": 0.0,
                        "weight_rna": 0.0,
                        "weight_ligand": 0.0,
                        "weight_motif": 0.0,
                        "weight_motif_related": 0.0,
                    }
                }
            },
        }
    )


@pytest.fixture
def small_model_config():
    return ArtiosBackboneConfig(
        diffusion_module=DiffusionModuleConfig(
            c_z=16,
            c_token=32,
            c_atom=32,
            c_s=32,
            transformer=DiffusionTransformerConfig(num_blocks=1, num_heads=4),
        )
    )


def test_main_train_loop_returns_positive_noise_level(
    backbone_configs, small_model_config, context, device
):
    model = ArtiosBackbone(backbone_configs, model_config=small_model_config).to(device)
    model.eval()

    with torch.no_grad():
        output = model.main_train_loop(context=context, seed=11)

    assert output.coordinate.shape == (1, 2, 16, 3)
    assert output.noise_level is not None
    assert output.noise_level.shape == (1, 2)
    assert (output.noise_level > 0).all()
    assert torch.isfinite(output.coordinate).all()


def test_backbone_model_config_is_immutable(small_model_config):
    with pytest.raises(FrozenInstanceError):
        small_model_config.n_model_seed = 2

    with pytest.raises(FrozenInstanceError):
        small_model_config.diffusion_module.c_token = 64


def test_backbone_runtime_config_is_independent_of_model_config(
    backbone_configs, small_model_config
):
    no_checkpointing = ArtiosBackboneRuntimeConfig(
        diffusion=DiffusionRuntimeConfig(checkpoint_grads=False)
    )
    with_checkpointing = ArtiosBackboneRuntimeConfig(
        diffusion=DiffusionRuntimeConfig(checkpoint_grads=True)
    )

    model_a = ArtiosBackbone(
        backbone_configs,
        model_config=small_model_config,
        runtime_config=no_checkpointing,
    )
    model_b = ArtiosBackbone(
        backbone_configs,
        model_config=small_model_config,
        runtime_config=with_checkpointing,
    )

    assert model_a.model_config == model_b.model_config
    assert model_a.runtime_config != model_b.runtime_config
    assert model_a.diffusion_module.diffusion_transformer.checkpoint_grads is False
    assert model_b.diffusion_module.diffusion_transformer.checkpoint_grads is True


def test_backbone_model_config_equality_tracks_architecture(small_model_config):
    deeper_model_config = ArtiosBackboneConfig(
        diffusion_module=DiffusionModuleConfig(
            c_z=16,
            c_token=32,
            c_atom=32,
            c_s=32,
            transformer=DiffusionTransformerConfig(num_blocks=2, num_heads=4),
        )
    )

    assert small_model_config != deeper_model_config


def test_backbone_rejects_legacy_architecture_config(
    backbone_configs, small_model_config
):
    legacy_configs = ConfigDict(backbone_configs.to_dict())
    legacy_configs.sigma_data = SIGMA_DATA
    with pytest.raises(ValueError, match="SIGMA_DATA"):
        ArtiosBackbone(legacy_configs, model_config=small_model_config)

    legacy_configs = ConfigDict(backbone_configs.to_dict())
    legacy_configs.model = ConfigDict({"diffusion_module": {"c_token": 64}})
    with pytest.raises(ValueError, match="DiffusionModuleConfig"):
        ArtiosBackbone(legacy_configs, model_config=small_model_config)

    legacy_configs = ConfigDict(backbone_configs.to_dict())
    legacy_configs.model = ConfigDict({"N_model_seed": 4})
    with pytest.raises(ValueError, match="ArtiosBackboneConfig"):
        ArtiosBackbone(legacy_configs, model_config=small_model_config)


def test_sigma_data_is_not_a_diffusion_config_field():
    with pytest.raises(TypeError):
        DiffusionConditioningConfig(sigma_data=SIGMA_DATA)

    with pytest.raises(TypeError):
        DiffusionConditioning(sigma_data=SIGMA_DATA)

    with pytest.raises(TypeError):
        DiffusionModuleConfig(sigma_data=SIGMA_DATA)

    with pytest.raises(TypeError):
        DiffusionModule(sigma_data=SIGMA_DATA)


def test_main_train_loop_uses_additive_edm_noise(
    backbone_configs, small_model_config, context, device, monkeypatch
):
    model = ArtiosBackbone(backbone_configs, model_config=small_model_config).to(device)
    recorder = RecordingDenoiser().to(device)
    model.diffusion_module = recorder
    model.noise_sampler = ConstantNoiseSampler(2.0)
    model.eval()

    coords = torch.arange(16 * 3, device=device, dtype=torch.float32).reshape(1, 16, 3)
    context.coordinate.coordinate = coords

    def identity_augmentation(x_input_coords, n_sample, mask):
        return x_input_coords.unsqueeze(1).expand(-1, n_sample, -1, -1).clone()

    monkeypatch.setattr(
        backbone_module, "centre_random_augmentation", identity_augmentation
    )

    with torch.no_grad():
        model.main_train_loop(context=context, seed=17)

    generator = torch.Generator(device=device).manual_seed(17)
    expected_clean = coords.unsqueeze(1).expand(-1, 2, -1, -1)
    expected_noise = (
        torch.randn(expected_clean.shape, device=device, generator=generator) * 2.0
    )

    torch.testing.assert_close(
        recorder.noise_level, torch.full((1, 2), 2.0, device=device)
    )
    torch.testing.assert_close(recorder.r_noisy, expected_clean + expected_noise)


def test_compute_loss_uses_edm_noise_level_weight(
    backbone_configs, small_model_config, context, device, monkeypatch
):
    model = ArtiosBackbone(backbone_configs, model_config=small_model_config).to(device)
    true = context.coordinate.coordinate
    pred = true.unsqueeze(1).expand(-1, 2, -1, -1) + 1.0
    noise_level = torch.tensor([[2.0, 4.0]], device=device)
    output = DiffusionOutput(coordinate=pred, noise_level=noise_level)

    monkeypatch.setattr(
        backbone_module,
        "align_to_prediction",
        lambda pred, true, mask, weight=None: true,
    )

    loss, _ = model.compute_loss(output, context)

    expected = (3.0 * edm_noise_level_scale(noise_level, sigma_data=SIGMA_DATA)).mean()
    torch.testing.assert_close(loss, expected)
