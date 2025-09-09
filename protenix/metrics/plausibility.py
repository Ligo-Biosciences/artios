from abc import abstractmethod
from typing import Any, List

from biotite.structure import AtomArray
from peppr import RamaScore, RotamerScore
from peppr.rotamer import ConformerClass


class PepprResidueScore:
    def __init__(self):
        self.num_valid_residues = 0.0
        self.num_total_residues = 0.0
        return

    @abstractmethod
    def compute_scores(self, pred_array: AtomArray) -> List[Any]:
        pass

    def update(self, pred_arrays: List[AtomArray]):
        """Compute the number of valid residues per sample.

        Args:
            pred_arrays: List of AtomArray objects.

        Returns:
            float: Number of valid residues per sample.
        """

        assert isinstance(
            pred_arrays, List
        ), f"Pred arrays must be a list, got {type(pred_arrays)}"
        assert isinstance(
            pred_arrays[0], AtomArray
        ), f"Pred array elements must be AtomArray, got {type(pred_arrays[0])}"

        for pred_array in pred_arrays:
            score_list = self.compute_scores(pred_array)

            assert isinstance(
                score_list, List
            ), f"Score list must be a list, got {type(score_list)}"

            self.num_total_residues += len(score_list)

            for residue in score_list:
                if residue.classification in [
                    ConformerClass.ALLOWED,
                    ConformerClass.FAVORED,
                ]:
                    self.num_valid_residues += 1.0

    def compute(self) -> float:
        assert (
            self.num_total_residues > 0.0
        ), "Number of total residues must be greater than 0"
        return self.num_valid_residues / self.num_total_residues

    def reset(self):
        self.num_valid_residues = 0.0
        self.num_total_residues = 0.0


class PerResidueRotamerScore(PepprResidueScore):
    def compute_scores(self, pred_array: AtomArray) -> List[Any]:
        return RotamerScore.from_atoms(pred_array)


class PerResidueRamaScore(PepprResidueScore):
    def compute_scores(self, pred_array: AtomArray) -> List[Any]:
        return RamaScore.from_atoms(pred_array)
