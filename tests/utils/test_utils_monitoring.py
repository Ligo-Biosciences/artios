import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from artios.utils.monitoring import (
    analyse_evaluation_latents,
    analyse_inference_latents,
    plot_accuracy_vs_timestep,
    plot_covariance_heatmap,
    plot_eigenvalues,
    plot_histograms,
    plot_qq_plots,
    plot_reliability_diagram,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestAnalyseEvaluationLatents:
    """Test suite for the analyse_evaluation_latents function."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def test_dimensions(self):
        """Test dimensions fixture."""
        return {
            "n_samples": 100,
            "n_tokens": 50,
            "c_latent": 8,
            "num_residues": 32,  # Based on STD_RESIDUES_AND_GAP
            "n_batches": 2,
        }

    @pytest.fixture
    def evaluation_test_data(self, device, test_dimensions):
        """Create test data for analyse_evaluation_latents function."""
        np.random.seed(42)

        dims = test_dimensions

        # Create multiple batches of data
        latents_list = []
        seq_logits_list = []
        latent_timestep_list = []
        restype_list = []
        is_protein_token_list = []
        is_rna_token_list = []
        is_dna_token_list = []
        is_ligand_token_list = []

        for batch_idx in range(dims["n_batches"]):
            batch_size = dims["n_samples"] * dims["n_tokens"]

            # Create latent data: [N_sample*N_token, c_latent]
            latents = np.random.randn(batch_size, dims["c_latent"]).astype(np.float32)

            # Create seq_logits data: [N_sample*N_token, NUM_RESIDUES]
            seq_logits = np.random.randn(batch_size, dims["num_residues"]).astype(
                np.float32
            )

            # Create timestep data: [N_sample*N_token]
            latent_timestep = np.random.uniform(0, 1, batch_size).astype(np.float32)

            # Create restype data: [N_sample*N_token] - random residue indices
            restype = np.random.randint(0, dims["num_residues"], batch_size).astype(
                np.int32
            )

            # Create entity type masks with mutually exclusive assignments
            entity_assignments = np.random.choice(
                4, batch_size
            )  # 0=protein, 1=rna, 2=dna, 3=ligand
            is_protein_token = entity_assignments == 0
            is_rna_token = entity_assignments == 1
            is_dna_token = entity_assignments == 2
            is_ligand_token = entity_assignments == 3

            latents_list.append(latents)
            seq_logits_list.append(seq_logits)
            latent_timestep_list.append(latent_timestep)
            restype_list.append(restype)
            is_protein_token_list.append(is_protein_token)
            is_rna_token_list.append(is_rna_token)
            is_dna_token_list.append(is_dna_token)
            is_ligand_token_list.append(is_ligand_token)

        return {
            "latents": latents_list,
            "seq_logits": seq_logits_list,
            "latent_timestep": latent_timestep_list,
            "restype": restype_list,
            "is_protein_token": is_protein_token_list,
            "is_rna_token": is_rna_token_list,
            "is_dna_token": is_dna_token_list,
            "is_ligand_token": is_ligand_token_list,
        }

    @pytest.fixture
    def empty_evaluation_data(self):
        """Create empty test data for edge case testing."""
        return {
            "latents": [],
            "seq_logits": [],
            "latent_timestep": [],
            "restype": [],
            "is_protein_token": [],
            "is_rna_token": [],
            "is_dna_token": [],
            "is_ligand_token": [],
        }

    def test_analyse_evaluation_latents_empty_data(self, empty_evaluation_data):
        """Test analyse_evaluation_latents with empty data."""
        result = analyse_evaluation_latents(empty_evaluation_data)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_analyse_evaluation_latents_valid_data(self, evaluation_test_data):
        """Test analyse_evaluation_latents with valid data."""
        result = analyse_evaluation_latents(evaluation_test_data)

        # Check that result is a dictionary
        assert isinstance(result, dict)

        # Check that we have plots for each entity type (accuracy and reliability)
        entity_types = ["protein", "rna", "dna", "ligand"]
        plot_types = ["accuracy_vs_timestep", "reliability"]

        for entity_type in entity_types:
            for plot_type in plot_types:
                key = f"{plot_type}_{entity_type}"
                if key in result:  # Some entity types might not have data
                    assert isinstance(
                        result[key], plt.Figure
                    ), f"Expected Figure for key: {key}"

        # Should have at least some plots
        assert len(result) > 0

        # Clean up
        for fig in result.values():
            plt.close(fig)

    def test_evaluation_latents_single_entity_type(self, device):
        """Test analyse_evaluation_latents with only one entity type."""
        single_entity_data = {
            "latents": [np.random.randn(100, 4).astype(np.float32)],
            "seq_logits": [np.random.randn(100, 32).astype(np.float32)],
            "latent_timestep": [np.random.uniform(0, 1, 100).astype(np.float32)],
            "restype": [np.random.randint(0, 32, 100).astype(np.int32)],
            "is_protein_token": [np.ones(100, dtype=bool)],  # All protein
            "is_rna_token": [np.zeros(100, dtype=bool)],
            "is_dna_token": [np.zeros(100, dtype=bool)],
            "is_ligand_token": [np.zeros(100, dtype=bool)],
        }

        result = analyse_evaluation_latents(single_entity_data)
        assert isinstance(result, dict)

        # Should have plots for protein only
        protein_keys = [k for k in result.keys() if "protein" in k]
        assert len(protein_keys) > 0

        # Clean up
        for fig in result.values():
            plt.close(fig)

    def test_evaluation_latents_perfect_accuracy(self, device):
        """Test analyse_evaluation_latents with perfect accuracy."""
        perfect_data = {
            "latents": [np.random.randn(100, 4).astype(np.float32)],
            "seq_logits": [np.eye(100, 32).astype(np.float32)],  # Perfect predictions
            "latent_timestep": [np.random.uniform(0, 1, 100).astype(np.float32)],
            "restype": [np.arange(100) % 32],  # Match with seq_logits
            "is_protein_token": [np.ones(100, dtype=bool)],
            "is_rna_token": [np.zeros(100, dtype=bool)],
            "is_dna_token": [np.zeros(100, dtype=bool)],
            "is_ligand_token": [np.zeros(100, dtype=bool)],
        }

        result = analyse_evaluation_latents(perfect_data)
        assert isinstance(result, dict)

        # Clean up
        for fig in result.values():
            plt.close(fig)


class TestAnalyseInferenceLatents:
    """Test suite for the analyse_inference_latents function."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def test_dimensions(self):
        """Test dimensions fixture."""
        return {
            "n_samples": 100,
            "n_tokens": 50,
            "c_latent": 8,
            "num_residues": 32,  # Based on STD_RESIDUES_AND_GAP
            "n_batches": 2,
        }

    @pytest.fixture
    def inference_test_data(self, device, test_dimensions):
        """Create test data for analyse_inference_latents function."""
        np.random.seed(42)

        dims = test_dimensions

        # Create multiple batches of latent data
        latents_list = []
        seq_logits_list = []

        for batch_idx in range(dims["n_batches"]):
            # Create latent data: [N_sample*N_token, c_latent]
            latents = np.random.randn(
                dims["n_samples"] * dims["n_tokens"], dims["c_latent"]
            ).astype(np.float32)

            # Create seq_logits data: [N_sample*N_token, NUM_RESIDUES]
            seq_logits = np.random.randn(
                dims["n_samples"] * dims["n_tokens"], dims["num_residues"]
            ).astype(np.float32)

            latents_list.append(latents)
            seq_logits_list.append(seq_logits)

        return {"latents": latents_list, "seq_logits": seq_logits_list}

    @pytest.fixture
    def empty_inference_data(self):
        """Create empty test data for edge case testing."""
        return {"latents": [], "seq_logits": []}

    def test_analyse_inference_latents_empty_data(self, empty_inference_data):
        """Test analyse_inference_latents with empty data."""
        result = analyse_inference_latents(empty_inference_data)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_analyse_inference_latents_valid_data(self, inference_test_data):
        """Test analyse_inference_latents with valid data."""
        result = analyse_inference_latents(inference_test_data)

        # Check that result is a dictionary
        assert isinstance(result, dict)

        # Check that all expected plots are generated
        expected_keys = [
            "hist_latents_batch_0",
            "qq_latents_batch_0",
            "cov_heatmap_latents_batch_0",
            "eigs_latents_batch_0",
            "hist_seq_logits_batch_0",
        ]

        for key in expected_keys:
            assert key in result, f"Missing expected key: {key}"
            assert isinstance(
                result[key], plt.Figure
            ), f"Expected Figure for key: {key}"

        # Clean up
        for fig in result.values():
            plt.close(fig)

    def test_analyse_inference_latents_single_batch(self, device):
        """Test analyse_inference_latents with single batch."""
        # Create single batch data
        single_batch_data = {
            "latents": [np.random.randn(100, 4).astype(np.float32)],
            "seq_logits": [np.random.randn(100, 32).astype(np.float32)],
        }

        result = analyse_inference_latents(single_batch_data)

        # Should have exactly one set of plots for batch_0
        expected_keys = [
            "hist_latents_batch_0",
            "qq_latents_batch_0",
            "cov_heatmap_latents_batch_0",
            "eigs_latents_batch_0",
            "hist_seq_logits_batch_0",
        ]

        assert len(result) == len(expected_keys)
        for key in expected_keys:
            assert key in result

        # Clean up
        for fig in result.values():
            plt.close(fig)

    def test_small_data_sizes(self, device):
        """Test functions with very small data sizes."""
        small_data = {
            "latents": [np.random.randn(10, 2).astype(np.float32)],
            "seq_logits": [np.random.randn(10, 32).astype(np.float32)],
        }

        result = analyse_inference_latents(small_data)
        assert isinstance(result, dict)

        # Clean up
        for fig in result.values():
            plt.close(fig)


class TestPlotFunctions:
    """Test suite for individual plotting functions."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def plot_test_data(self, device):
        """Create test data for plotting functions."""
        np.random.seed(42)
        return {
            "test_data": np.random.randn(100, 4).astype(np.float32),
            "test_data_large": np.random.randn(100, 25).astype(np.float32),
            "timesteps": np.random.uniform(0, 1, 100).astype(np.float32),
            "accuracy": np.random.uniform(0, 1, 100).astype(np.float32),
            "confidence": np.random.uniform(0, 1, 100).astype(np.float32),
        }

    def test_plot_accuracy_vs_timestep(self, plot_test_data):
        """Test plot_accuracy_vs_timestep function."""
        data = plot_test_data
        fig = plot_accuracy_vs_timestep(data["timesteps"], data["accuracy"], "protein")

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

        # Check that title contains entity type
        ax = fig.axes[0]
        assert "Protein" in ax.get_title()

        plt.close(fig)

    def test_plot_reliability_diagram(self, plot_test_data):
        """Test plot_reliability_diagram function."""
        data = plot_test_data
        fig = plot_reliability_diagram(
            data["timesteps"], data["confidence"], data["accuracy"], "rna"
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

        # Check that title contains entity type
        ax = fig.axes[0]
        assert "Rna" in ax.get_title()

        plt.close(fig)

    def test_plot_histograms(self, plot_test_data):
        """Test plot_histograms function."""
        data = plot_test_data
        fig = plot_histograms(data["test_data"], "_test", mode="latents")

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4  # Should have at least 4 subplots for 4 dimensions

        plt.close(fig)

    def test_plot_histograms_seq_logits_mode(self, plot_test_data):
        """Test plot_histograms with seq_logits mode."""
        data = plot_test_data
        fig = plot_histograms(data["test_data"], "_test", mode="seq_logits")

        assert isinstance(fig, plt.Figure)
        # Check that the title contains "seq_logits"
        assert "seq_logits" in fig._suptitle.get_text()

        plt.close(fig)

    def test_plot_qq_plots(self, plot_test_data):
        """Test plot_qq_plots function."""
        data = plot_test_data
        fig = plot_qq_plots(data["test_data"], "_test")

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4

        plt.close(fig)

    def test_plot_covariance_heatmap(self, plot_test_data):
        """Test plot_covariance_heatmap function."""
        data = plot_test_data
        fig = plot_covariance_heatmap(data["test_data"], "_test")

        assert isinstance(fig, plt.Figure)
        # Heatmap has main axis + colorbar axis
        assert len(fig.axes) == 2

        plt.close(fig)

    def test_plot_eigenvalues(self, plot_test_data):
        """Test plot_eigenvalues function."""
        data = plot_test_data
        fig = plot_eigenvalues(data["test_data"], "_test")

        assert isinstance(fig, plt.Figure)
        # Should have 2 y-axes (eigenvalues and cumulative variance)
        assert len(fig.axes) == 2

        plt.close(fig)


class TestIntegrationWithFiles:
    """Integration tests that save figures to files."""

    @pytest.fixture
    def temp_dir(self):
        """Create and cleanup temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def file_test_evaluation_data(self):
        """Create test data for file integration tests."""
        np.random.seed(42)
        return {
            "latents": [np.random.randn(100, 4).astype(np.float32)],
            "seq_logits": [np.random.randn(100, 32).astype(np.float32)],
            "latent_timestep": [np.random.uniform(0, 1, 100).astype(np.float32)],
            "restype": [np.random.randint(0, 32, 100).astype(np.int32)],
            "is_protein_token": [
                np.random.choice([0, 1], 100, p=[0.5, 0.5]).astype(bool)
            ],
            "is_rna_token": [np.random.choice([0, 1], 100, p=[0.8, 0.2]).astype(bool)],
            "is_dna_token": [np.random.choice([0, 1], 100, p=[0.8, 0.2]).astype(bool)],
            "is_ligand_token": [
                np.random.choice([0, 1], 100, p=[0.9, 0.1]).astype(bool)
            ],
        }

    @pytest.fixture
    def file_test_inference_data(self):
        """Create test data for file integration tests."""
        np.random.seed(42)
        return {
            "latents": [np.random.randn(100, 4).astype(np.float32)],
            "seq_logits": [np.random.randn(100, 32).astype(np.float32)],
        }

    def test_save_evaluation_figures_to_files(
        self, temp_dir, file_test_evaluation_data
    ):
        """Test that evaluation analysis figures can be saved to files."""
        # Generate figures
        result = analyse_evaluation_latents(file_test_evaluation_data)

        # Save each figure to a file
        saved_files = []
        for name, fig in result.items():
            file_path = Path(temp_dir) / f"eval_{name}.png"
            fig.savefig(file_path, dpi=150, bbox_inches="tight")
            saved_files.append(file_path)
            plt.close(fig)

        # Verify files were created and have reasonable size
        for file_path in saved_files:
            assert file_path.exists()
            assert file_path.stat().st_size > 1000  # At least 1KB

    def test_save_inference_figures_to_files(self, temp_dir, file_test_inference_data):
        """Test that inference analysis figures can be saved to files."""
        # Generate figures
        result = analyse_inference_latents(file_test_inference_data)

        # Save each figure to a file
        saved_files = []
        for name, fig in result.items():
            file_path = Path(temp_dir) / f"inf_{name}.png"
            fig.savefig(file_path, dpi=150, bbox_inches="tight")
            saved_files.append(file_path)
            plt.close(fig)

        # Verify files were created and have reasonable size
        for file_path in saved_files:
            assert file_path.exists()
            assert file_path.stat().st_size > 1000  # At least 1KB


if __name__ == "__main__":
    pytest.main([__file__])
