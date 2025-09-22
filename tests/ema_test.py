import numpy as np
import pytest
import torch
import torch.nn as nn

from protenix.utils.ema_utils import p_dot_p, solve_weights
from runner.ema import PowerLawEMAWrapper


class TestEMA:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_p_dot_p(self):
        t_a = [1000, 2000, 3000]
        t_b = t_a

        gamma_a = [7.0, 8.0]
        gamma_b = [6.0]

        # Convert inputs to numpy arrays
        t_a = np.array([t_a]).reshape(1, -1)  # [1, len(t_a)]
        gamma_a = np.array(gamma_a).reshape(-1, 1)  # [len(gamma_a), 1]
        t_b = np.array([t_b]).reshape(1, -1)  # [1, len(t_b)]
        gamma_b = np.array(gamma_b).reshape(-1, 1)  # [len(gamma_b), 1]

        # Broadcast arrays to match shapes
        t_a, gamma_a = np.broadcast_arrays(t_a, gamma_a)
        t_b, gamma_b = np.broadcast_arrays(t_b, gamma_b)

        # Convert to column/row vectors
        def rv(x):
            """Convert to column vector."""
            return np.float64(x).reshape(-1, 1)

        def cv(x):
            """Convert to row vector."""
            return np.float64(x).reshape(1, -1)

        # Calculate matrices A and B
        t_a_col = rv(t_a)
        gamma_a_col = rv(gamma_a)
        t_b_row = cv(t_b)
        gamma_b_row = cv(gamma_b)

        A = p_dot_p(t_a_col, gamma_a_col, t_b_row, gamma_b_row)

        assert not np.any(np.isnan(A))
        assert not np.any(np.isinf(A))

    def test_solve_weights(self):
        t_i = [1000, 2000, 3000]
        gamma_i = [7.0, 8.0]
        gamma_r = [6.0, 7.0, 8.0]

        weights = solve_weights(t_i, gamma_i, gamma_r)

        print("weights shape", weights.shape)
        print("weights", weights)

        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))


class TestPowerLawEMAWrapper:
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        return model

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_register(self, simple_model):
        """Test registering model parameters."""
        wrapper = PowerLawEMAWrapper(
            simple_model,
            sigma=10.0,
            mutable_param_keywords=["0.weight", "0.bias", "2.weight", "2.bias"],
        )
        wrapper.register()

        # Check that all parameters are registered
        for name, param in simple_model.named_parameters():
            assert name in wrapper.shadow
            assert torch.equal(wrapper.shadow[name], param.data)

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
