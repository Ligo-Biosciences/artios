import pytest
import torch

from artios.metrics.sequence_metrics import SequenceMetrics, compute_sequence_recovery


class TestSequenceMetrics:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def sequence_metrics(self, device):
        """Initialize SequenceMetrics instance"""
        return SequenceMetrics().to(device)

    @pytest.fixture
    def sample_data(self, device):
        """Create sample data for testing"""
        n_samples = 4
        n_tokens = 8
        n_restype = 21  # 20 amino acids + 1 for unknown

        # Create seq_logits with some predictable patterns
        seq_logits = torch.randn(n_samples, n_tokens, n_restype, device=device)

        # Create one-hot encoded true sequence
        true_indices = torch.randint(0, n_restype, size=(n_tokens,), device=device)
        true_restype = torch.zeros(n_tokens, n_restype, device=device)
        true_restype.scatter_(-1, true_indices.unsqueeze(-1), 1.0)

        # Create a case where some predictions are correct
        # Make first sample's first token prediction match the true value
        seq_logits[0, 0, true_indices[0]] = 10.0  # High logit for correct answer
        seq_logits[0, 0, :] = (
            seq_logits[0, 0, :] - seq_logits[0, 0, true_indices[0]] + 10.0
        )

        # Create token mask (all valid)
        token_mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        pred_dict = {"seq_logits": seq_logits}
        input_feature_dict = {"restype": true_restype, "token_mask": token_mask}

        return pred_dict, input_feature_dict, true_indices

    def test_sequence_metrics_initialization(self, sequence_metrics):
        """Test that SequenceMetrics initializes correctly"""
        assert isinstance(sequence_metrics, SequenceMetrics)
        assert hasattr(sequence_metrics, "compute_metrics")
        assert hasattr(sequence_metrics, "aggregate")
        assert hasattr(sequence_metrics, "aggregate_sequence_metrics")

    def test_aggregate_basic(self, sequence_metrics, device):
        """Test basic functionality of aggregate method"""
        n_samples = 6

        # Create test values
        vals = torch.rand(n_samples, device=device)

        aggregated = sequence_metrics.aggregate(vals)

        # Check that all expected aggregators are present
        expected_keys = ["best", "worst", "random", "mean", "median"]
        for key in expected_keys:
            assert key in aggregated

        # Check output shapes (should remove the sample dimension)
        for key, value in aggregated.items():
            assert value.shape.numel() == 1

        # Check logical relationships
        assert torch.all(aggregated["best"] >= aggregated["worst"])
        assert torch.all(aggregated["best"] >= aggregated["mean"])
        assert torch.all(aggregated["mean"] >= aggregated["worst"])

    def test_aggregate_custom_aggregators(self, sequence_metrics, device):
        """Test aggregate with custom aggregators"""
        n_samples = 4

        vals = torch.rand(n_samples, device=device)

        custom_aggregators = {
            "custom_max": lambda x: x.max(dim=-1)[0],
            "custom_sum": lambda x: x.sum(dim=-1),
        }

        aggregated = sequence_metrics.aggregate(vals, aggregators=custom_aggregators)

        # Check that custom aggregators are included
        assert "custom_max" in aggregated
        assert "custom_sum" in aggregated

        # Check that basic aggregators are still present
        assert "best" in aggregated
        assert "mean" in aggregated


class TestComputeSequenceRecovery:
    """Test the standalone compute_sequence_recovery function"""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_compute_sequence_recovery_basic(self, device):
        """Test basic functionality of compute_sequence_recovery function"""
        n_samples = 3
        n_tokens = 5
        n_restype = 21

        seq_logits = torch.randn(n_samples, n_tokens, n_restype, device=device)
        true_indices = torch.randint(0, n_restype, size=(n_tokens,), device=device)
        true_restype = torch.zeros(n_tokens, n_restype, device=device)
        true_restype.scatter_(-1, true_indices.unsqueeze(-1), 1.0)
        token_mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        pred_dict = {"seq_logits": seq_logits}
        input_feature_dict = {"restype": true_restype, "token_mask": token_mask}

        recovery_rate = compute_sequence_recovery(pred_dict, input_feature_dict)
        print("recovery_rate", recovery_rate)

        assert recovery_rate.shape.numel() == n_samples
        assert torch.all(recovery_rate >= 0.0)
        assert torch.all(recovery_rate <= 1.0)
        assert not torch.isnan(recovery_rate).any()


if __name__ == "__main__":
    pytest.main([__file__])
