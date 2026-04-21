"""Tests for motif_parser.py."""

from pathlib import Path

import pytest

from artios.data.parsing.motif import Motif

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "motifs"


class TestMotif:
    """Tests for Motif dataclass."""

    @pytest.fixture
    def ccd_input_path(self):
        return FIXTURES_DIR / "motif_input.json"

    @pytest.fixture
    def smiles_input_path(self):
        return FIXTURES_DIR / "motif_input_smiles.json"

    def test_from_spec_file_not_found(self, tmp_path):
        """Test that missing CIF file raises FileNotFoundError."""
        from artios.data.parsing.json import CIFFile, MotifSpec

        motif_spec = MotifSpec(cif_file=CIFFile(path=tmp_path / "nonexistent.cif"))

        with pytest.raises(FileNotFoundError):
            Motif.from_spec(motif_spec)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
