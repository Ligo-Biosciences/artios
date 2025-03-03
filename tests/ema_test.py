import pytest
import torch
import numpy as np

from runner.ema import EMAWrapper, PowerLawEMAWrapper, p_dot_p, solve_weights, beta_power_law


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
        rv = lambda x: np.float64(x).reshape(-1, 1)  # Convert to column vector
        cv = lambda x: np.float64(x).reshape(1, -1)  # Convert to row vector

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
        

if __name__ == "__main__":
    pytest.main([__file__])
