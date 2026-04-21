import pytest
import torch
import torch.nn as nn

from runner.ema import PowerLawEMAWrapper


class TestPowerLawEMAWrapper:
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        return model

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_update_single_step(self, simple_model):
        """Test single update step."""
        wrapper = PowerLawEMAWrapper(
            simple_model,
            sigma=10.0,
            mutable_param_keywords=["0.weight", "0.bias", "2.weight", "2.bias"],
        )
        wrapper.register()

        # Store initial weights
        initial_weights = {
            name: param.data.clone() for name, param in simple_model.named_parameters()
        }

        # Modify model weights
        for param in simple_model.parameters():
            param.data += 0.1

        # Update EMA
        wrapper.update(t=2)

        # Check that shadow weights have been updated
        for name in wrapper.shadow:
            assert not torch.equal(wrapper.shadow[name], initial_weights[name])

    def test_update_multiple_steps(self, simple_model):
        """Test multiple update steps."""
        wrapper = PowerLawEMAWrapper(
            simple_model,
            sigma=5.0,
            mutable_param_keywords=["0.weight", "0.bias", "2.weight", "2.bias"],
        )
        wrapper.register()

        # Perform multiple updates
        for t in range(1, 101):
            # Slightly modify weights
            for param in simple_model.parameters():
                param.data += torch.randn_like(param.data) * 0.01

            # Update EMA
            wrapper.update(t=t)

        # Verify shadow weights exist and are finite
        for name in wrapper.shadow:
            assert torch.isfinite(wrapper.shadow[name]).all()

    def test_apply_shadow_and_restore(self, simple_model):
        """Test applying shadow weights and restoring original."""
        wrapper = PowerLawEMAWrapper(
            simple_model,
            sigma=10.0,
            mutable_param_keywords=["0.weight", "0.bias", "2.weight", "2.bias"],
        )
        wrapper.register()

        # Store original weights
        original_weights = {
            name: param.data.clone() for name, param in simple_model.named_parameters()
        }

        # Modify shadow weights
        for name in wrapper.shadow:
            wrapper.shadow[name] += 0.5

        # Apply shadow weights
        wrapper.apply_shadow()

        # Check that model weights now match shadow weights
        for name, param in simple_model.named_parameters():
            assert torch.equal(param.data, wrapper.shadow[name])
            assert name in wrapper.backup

        # Restore original weights
        wrapper.restore()

        # Check that weights are restored
        for name, param in simple_model.named_parameters():
            assert torch.equal(param.data, original_weights[name])
        assert wrapper.backup == {}


if __name__ == "__main__":
    pytest.main([__file__])
