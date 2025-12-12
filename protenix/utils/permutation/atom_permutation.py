from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch import Tensor

from protenix.metrics.rmsd import rmsd, self_aligned_rmsd
from protenix.model.utils import expand_at_dim, pad_at_dim
from protenix.utils.logger import get_logger
from protenix.utils.permutation.utils import Checker, save_permutation_error
from protenix.utils.typing import Bool, Float, Int, typecheck

logger = get_logger(__name__)


# =============================================================================
# Data Structures for Symmetric Residue Collection
# =============================================================================


@dataclass(frozen=True)
class ResiduePermutationInfo:
    """Symmetry information for a single residue.

    Attributes:
        start: Start index (inclusive) of this residue's atoms in the full coordinate array.
        end: End index (exclusive) of this residue's atoms in the full coordinate array.
        coords: Coordinates of atoms in this residue. Shape: [n_atoms, 3].
        coord_mask: Boolean mask indicating resolved atoms. Shape: [n_atoms].
        permutations: Valid permutations for this residue. Shape: [n_perms, n_atoms].
            The first permutation (index 0) is always the identity.
    """

    start: int
    end: int
    coords: Float[torch.Tensor, "n_atoms 3"]
    coord_mask: Bool[torch.Tensor, "n_atoms"]
    permutations: Int[torch.Tensor, "n_perms n_atoms"]

    @property
    def n_atoms(self) -> int:
        return self.end - self.start

    @property
    def n_permutations(self) -> int:
        return self.permutations.size(0)


@dataclass
class SymmetricResidueCollection:
    """Collection of residues requiring symmetric atom correction.

    Attributes:
        residues: List of residues with valid symmetric permutations.
        total_residue_count: Total number of residues in the structure
            (including those without symmetry).
    """

    residues: list[ResiduePermutationInfo]
    total_residue_count: int

    def __len__(self) -> int:
        return len(self.residues)

    def __iter__(self) -> Iterator[ResiduePermutationInfo]:
        return iter(self.residues)

    def is_empty(self) -> bool:
        return len(self.residues) == 0

    @property
    def positions(self) -> list[tuple[int, int]]:
        """Return (start, end) positions for backward compatibility."""
        return [(r.start, r.end) for r in self.residues]

    @property
    def coord_list(self) -> list[torch.Tensor]:
        """Return coordinate tensors for backward compatibility."""
        return [r.coords for r in self.residues]

    @property
    def coord_mask_list(self) -> list[torch.Tensor]:
        """Return coordinate mask tensors for backward compatibility."""
        return [r.coord_mask for r in self.residues]

    @property
    def perm_list(self) -> list[torch.Tensor]:
        """Return permutation tensors for backward compatibility."""
        return [r.permutations for r in self.residues]


# =============================================================================
# Result Data Structures
# =============================================================================


@dataclass
class PermutationStats:
    """Statistics from atom permutation optimization."""

    n_residues: int
    n_residues_with_symmetry: int
    n_residues_permuted: float
    has_residue_permuted: float
    unpermuted_rmsd: Optional[float] = None
    permuted_rmsd: Optional[float] = None

    @property
    def improved_rmsd(self) -> Optional[float]:
        """RMSD improvement from permutation (positive = better)."""
        if self.unpermuted_rmsd is None or self.permuted_rmsd is None:
            return None
        return self.unpermuted_rmsd - self.permuted_rmsd


@dataclass
class AtomPermutationResult:
    """Result from AtomPermutation.__call__."""

    permutation: Int[Tensor, "*batch n"]
    stats: PermutationStats


@dataclass
class PermutedLabel:
    """Result when permuting ground truth labels.

    For batched inputs (run_batched), stats is a list with one entry per batch element.
    """

    coordinate: Float[Tensor, "*batch n 3"]
    coordinate_mask: Bool[Tensor, "*batch n"]
    permutation: Int[Tensor, "*batch n"]
    stats: PermutationStats | list[PermutationStats]


@dataclass
class PermutedPrediction:
    """Result when permuting predictions.

    For batched inputs (run_batched), stats is a list with one entry per batch element.
    """

    coordinate: Float[Tensor, "*batch n 3"]
    permutation: Int[Tensor, "*batch n"]
    stats: PermutationStats | list[PermutationStats]


PermutationResult = PermutedLabel | PermutedPrediction


# =============================================================================
# Helper Functions for Residue Boundary Detection and Permutation Building
# =============================================================================


@typecheck
def _find_residue_boundaries(
    ref_space_uid: Int[torch.Tensor, "n_atoms"],
) -> list[tuple[int, int]]:
    """Find (start, end) index pairs for each residue based on unique IDs.

    Args:
        ref_space_uid: Unique ID for each atom's residue. Atoms belonging to the
            same residue have the same ID. IDs must be contiguous (all atoms of
            a residue are adjacent).

    Returns:
        List of (start, end) tuples where start is inclusive and end is exclusive.
    """
    n_atoms = ref_space_uid.size(0)
    if n_atoms == 0:
        return []

    # Find positions where residue ID changes
    # Note: We create this on the same device as input for consistency
    device = ref_space_uid.device
    id_changes = ref_space_uid[1:] != ref_space_uid[:-1]
    change_positions = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.nonzero(id_changes, as_tuple=True)[0] + 1,
            torch.tensor([n_atoms], device=device),
        ]
    )

    # Convert to list of (start, end) tuples
    boundaries = [
        (change_positions[i].item(), change_positions[i + 1].item())
        for i in range(len(change_positions) - 1)
    ]

    return boundaries


@typecheck
def _build_residue_permutation_info(
    start: int,
    end: int,
    coord: Float[torch.Tensor, "n_total_atoms 3"],
    coord_mask: Bool[torch.Tensor, "n_total_atoms"],
    atom_perm_list: list[list[int]],
    device: torch.device,
    run_checker: bool = False,
) -> ResiduePermutationInfo | None:
    """Build permutation info for a single residue.

    Args:
        start: Start index (inclusive) of this residue in the atom arrays.
        end: End index (exclusive) of this residue in the atom arrays.
        coord: Full coordinate array for all atoms.
        coord_mask: Full mask array for all atoms.
        atom_perm_list: Per-atom permutation list for all atoms.
        device: Device to create tensors on.
        run_checker: If True, run additional validity checks.

    Returns:
        ResiduePermutationInfo if this residue has valid permutations, None otherwise.
        Returns None if:
        - The residue has fewer than 3 resolved atoms (can't align)
        - All permutations are identity (no symmetry to correct)
    """
    res_coord_mask = coord_mask[start:end].bool()  # [n_res_atoms]
    n_res_atoms = end - start

    # Skip if fewer than 3 resolved atoms (alignment requires at least 3)
    if res_coord_mask.sum() < 3:
        return None

    # Build permutation tensor and deduplicate
    perm = torch.tensor(atom_perm_list[start:end], device=device, dtype=torch.long)
    perm = torch.unique(perm, dim=-1)  # [n_res_atoms, n_perms]
    # n_perms = perm.size(-1)

    # Basic validity checks
    assert (
        perm.min().item() == 0
    ), f"Permutation indices must start at 0, got {perm.min()}"
    assert (
        perm.max().item() == n_res_atoms - 1
    ), f"Permutation indices must be < n_atoms ({n_res_atoms}), got {perm.max()}"

    # Development checks
    if run_checker:
        Checker.are_permutations(perm, dim=0)
        Checker.contains_identity(perm, dim=0)

    # Filter out permutations where all symmetric atoms are unresolved
    identity_perm = torch.arange(n_res_atoms, device=device).unsqueeze(
        dim=-1
    )  # [n_res_atoms, 1]
    is_symmetric_atom = perm != identity_perm  # [n_res_atoms, n_perms]
    is_symmetric_atom_resolved = is_symmetric_atom * res_coord_mask.unsqueeze(dim=-1)
    is_valid_perm = is_symmetric_atom_resolved.any(dim=0)  # [n_perms]

    if not is_valid_perm.any():
        # No valid permutation other than identity
        return None

    # Keep only valid permutations and prepend identity
    perm = perm[..., is_valid_perm]
    perm = torch.cat([identity_perm, perm], dim=-1)  # Identity first
    perm = perm.transpose(-1, -2)  # [n_perms, n_res_atoms]

    return ResiduePermutationInfo(
        start=start,
        end=end,
        coords=coord[start:end, :],
        coord_mask=res_coord_mask,
        permutations=perm,
    )


@typecheck
def collect_residues_with_symmetric_atoms(
    coord: Float[torch.Tensor, "n_atoms 3"],
    coord_mask: Bool[torch.Tensor, "n_atoms"],
    ref_space_uid: Int[torch.Tensor, "n_atoms"],
    atom_perm_list: list[list[int]],
    run_checker: bool = False,
) -> SymmetricResidueCollection:
    """Convert atom-level permutation attributes to residue-level attributes.

    Only residues that require symmetric corrections are returned. A residue is
    included if it has:
    - At least 3 resolved atoms (required for alignment)
    - At least one non-identity permutation with resolved symmetric atoms

    Args:
        coord: Coordinates of atoms.
        coord_mask: Mask indicating whether each atom is resolved in ground truth.
        ref_space_uid: Unique ID for each unique conformer.
            Atoms of the same residue must be contiguous.  # TODO: make sure motif stuff abides by this.
        atom_perm_list: Per-atom permutation list. Each sublist contains the
            permutation indices for that atom across all valid permutations.
            len(atom_perm_list) == n_atoms.
        run_checker: If True, run additional validity checks.

    Returns:
        SymmetricResidueCollection containing residues with valid permutations.
    """
    device = coord_mask.device

    # Find residue boundaries
    boundaries = _find_residue_boundaries(ref_space_uid)
    n_res = len(boundaries)

    # Sanity check: number of boundaries should match unique residue IDs
    assert n_res == len(torch.unique(ref_space_uid)), (
        f"Residue boundaries ({n_res}) don't match unique ref_space_uids "
        f"({len(torch.unique(ref_space_uid))}). Are atoms contiguous per residue?"
    )

    # Build permutation info for each residue, filtering invalid ones
    residues: list[ResiduePermutationInfo] = []
    for start, end in boundaries:
        info = _build_residue_permutation_info(
            start=start,
            end=end,
            coord=coord,
            coord_mask=coord_mask,
            atom_perm_list=atom_perm_list,
            device=device,
            run_checker=run_checker,
        )
        if info is not None:
            residues.append(info)

    return SymmetricResidueCollection(
        residues=residues,
        total_residue_count=n_res,
    )


@dataclass(frozen=True)
class PermutedResidueCoords:
    """Result of applying all permutations to residue coordinates.

    This structure holds the padded, concatenated coordinates and masks
    for all permutations across all residues.

    Attributes:
        coords: Permuted coordinates for all residues, padded to max atom count.
            Shape: [n_total_perms, max_n_atoms, 3].
        coord_mask: Mask indicating valid (non-padded) atoms.
            Shape: [n_total_perms, max_n_atoms].
        max_n_atoms: Maximum number of atoms across all residues.
    """

    coords: Float[torch.Tensor, "n_total_perms max_n_atoms 3"]
    coord_mask: Bool[torch.Tensor, "n_total_perms max_n_atoms"]
    max_n_atoms: int

    @property
    def n_total_perms(self) -> int:
        """Total number of permutations across all residues."""
        return self.coords.size(0)


def _apply_permutations_to_residue(
    residue: ResiduePermutationInfo,
    max_n_atoms: int,
    run_checker: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply all permutations to a single residue's coordinates.

    Args:
        residue: The residue permutation info containing coords, mask, and permutations.
        max_n_atoms: The maximum atom count to pad to.
        run_checker: If True, run additional validity checks.

    Returns:
        Tuple of (permuted_coords, permuted_mask):
            - permuted_coords: [n_perms, max_n_atoms, 3]
            - permuted_mask: [n_perms, max_n_atoms]
    """
    perm = residue.permutations  # [n_perms, n_atoms]
    n_atoms = residue.n_atoms

    # Apply permutations via advanced indexing
    res_coord_permuted = residue.coords[perm]  # [n_perms, n_atoms, 3]
    res_coord_mask_permuted = residue.coord_mask[perm]  # [n_perms, n_atoms]

    if run_checker:
        Checker.are_permutations(perm, dim=-1)
        Checker.batch_permute(perm, residue.coords, res_coord_permuted)
        Checker.batch_permute(perm, residue.coord_mask, res_coord_mask_permuted)

    # Pad to max_n_atoms if needed
    if n_atoms < max_n_atoms:
        pad_length = (0, max_n_atoms - n_atoms)
        res_coord_permuted = pad_at_dim(
            res_coord_permuted, dim=-2, pad_length=pad_length
        )  # [n_perms, max_n_atoms, 3]
        res_coord_mask_permuted = pad_at_dim(
            res_coord_mask_permuted, dim=-1, pad_length=pad_length
        )  # [n_perms, max_n_atoms]

    return res_coord_permuted, res_coord_mask_permuted


def collect_permuted_coords(
    residues: list[ResiduePermutationInfo],
    run_checker: bool = False,
) -> PermutedResidueCoords:
    """Apply all permutations to coordinates from a list of residues.

    For each residue, applies all valid permutations to the coordinates and mask.
    Results are padded to the maximum atom count and concatenated.

    Args:
        residues: List of residues with symmetric permutations.
        run_checker: If True, run additional validity checks.

    Returns:
        PermutedResidueCoords containing the batched permuted coordinates and masks.

    Raises:
        ValueError: If the residue list is empty.
    """
    if not residues:
        raise ValueError("Cannot collect permuted coords from empty residue list")

    # Find max atom count across all residues
    max_n_atoms = max(r.n_atoms for r in residues)

    # Apply permutations to each residue
    permuted_coords_list: list[torch.Tensor] = []
    permuted_masks_list: list[torch.Tensor] = []

    for residue in residues:
        coords, mask = _apply_permutations_to_residue(
            residue=residue,
            max_n_atoms=max_n_atoms,
            run_checker=run_checker,
        )
        permuted_coords_list.append(coords)
        permuted_masks_list.append(mask)

    # Concatenate all permutations
    all_coords = torch.cat(permuted_coords_list, dim=0)
    all_masks = torch.cat(permuted_masks_list, dim=0)

    return PermutedResidueCoords(
        coords=all_coords,
        coord_mask=all_masks,
        max_n_atoms=max_n_atoms,
    )


class AtomPermutation:
    """Optimizer for symmetric atom permutations in molecular structures.

    Finds optimal permutations of symmetric atoms (e.g., in symmetric residues)
    to minimize RMSD between predicted and ground truth coordinates.

    Args:
        eps: Small number for numerical stability in alignment.
        run_checker: If True, run additional validity checks.
        global_align_wo_symmetric_atom: If True, exclude symmetric atoms from
            the global alignment step.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        run_checker: bool = False,
        global_align_wo_symmetric_atom: bool = False,
    ):
        self.eps = eps
        self.run_checker = run_checker
        self.global_align_wo_symmetric_atom = global_align_wo_symmetric_atom

    @staticmethod
    @typecheck
    def global_align_pred_to_true(
        pred_coord: Float[Tensor, "*batch n 3"],
        true_coord: Float[Tensor, "n 3"],
        true_coord_mask: Bool[Tensor, "n"],
        eps: float = 1e-8,
    ) -> tuple[Float[Tensor, "*batch"], Float[Tensor, "*batch n 3"]]:
        """Align predicted coordinates to true coordinates via Kabsch algorithm.

        Args:
            pred_coord: Predicted coordinates, optionally batched.
            true_coord: Ground truth coordinates (unbatched).
            true_coord_mask: Mask indicating which atoms to include.
            eps: Small tolerance to avoid floating point issues.

        Returns:
            Tuple of (aligned_rmsd, transformed_pred_coord).
        """
        batch_dims = pred_coord.dim() - true_coord.dim()
        if batch_dims > 0:
            batch_size = pred_coord.size(0)
            true_coord = expand_at_dim(true_coord, dim=0, n=batch_size)
            true_coord_mask = expand_at_dim(true_coord_mask, dim=0, n=batch_size)

        with torch.amp.autocast("cuda", enabled=False):
            aligned_rmsd, transformed_pred_coord, _, _ = self_aligned_rmsd(
                pred_pose=pred_coord.float(),
                true_pose=true_coord.float(),
                atom_mask=true_coord_mask,
                allowing_reflection=False,
                reduce=False,
                eps=eps,
            )

        return aligned_rmsd, transformed_pred_coord

    @staticmethod
    def get_identity_permutation(
        batch_shape: tuple[int, ...],
        n_atoms: int,
        device: torch.device,
    ) -> Int[Tensor, "*batch n"]:
        """Return identity permutation indices.

        Args:
            batch_shape: Shape of batch dimensions (empty tuple or single int).
            n_atoms: Number of atoms.
            device: Device to create tensor on.

        Returns:
            Identity permutation of shape [n_atoms] or [batch, n_atoms].
        """
        identity = torch.arange(n_atoms, device=device)
        if len(batch_shape) == 0:
            return identity
        assert (
            len(batch_shape) == 1
        ), f"batch_shape must be 0 or 1 dim, got {batch_shape}"
        batch_size = batch_shape[0]
        return torch.stack([identity] * batch_size, dim=0)

    @staticmethod
    def _optimize_per_residue_permutation_by_rmsd(
        pred_coords_list: list[torch.Tensor],
        residues: list[ResiduePermutationInfo],
        eps: float = 1e-8,
        run_checker: bool = False,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Find the optimal permutations to minimize RMSD between pred and true coordinates.

        For each residue, tries all valid permutations and selects the one that
        minimizes the RMSD between the (permuted) ground truth and the prediction.

        Args:
            pred_coords_list: List of predicted coordinates per residue.
                Each element has shape [n_atoms, 3] or [batch, n_atoms, 3].
            residues: List of residues containing ground truth coords, masks,
                and valid permutations.
            eps: Small number for numerical stability in RMSD calculation.
            run_checker: If True, run additional validity checks.

        Returns:
            Tuple of four lists, one element per residue:
            - best_permutation_list: Optimal permutation indices.
                Shape per element: [n_atoms] or [batch, n_atoms].
            - is_permuted_list: Whether non-identity permutation was selected.
                Shape per element: [] or [batch].
            - optimized_rmsd_list: RMSD after applying optimal permutation.
                Shape per element: [] or [batch].
            - original_rmsd_list: RMSD with identity permutation (no swap).
                Shape per element: [] or [batch].
        """
        assert len(pred_coords_list) == len(
            residues
        ), f"Mismatch: {len(pred_coords_list)} pred coords vs {len(residues)} residues"

        # Collect permuted ground truth coordinates
        permuted = collect_permuted_coords(residues, run_checker=run_checker)
        max_n_atoms = permuted.max_n_atoms
        n_total_perms = permuted.n_total_perms

        # Get per-residue permutation counts for indexing later
        per_residue_n_perms = [r.n_permutations for r in residues]

        # Pad predicted coordinates to max_n_atoms and repeat for each permutation
        padded_pred_coords = [
            pad_at_dim(p, dim=-2, pad_length=(0, max_n_atoms - p.size(-2)))
            for p in pred_coords_list
        ]
        # Repeat each pred coord tensor n_perms times
        pred_coord = torch.stack(
            sum(
                [
                    [p] * n_perms
                    for n_perms, p in zip(per_residue_n_perms, padded_pred_coords)
                ],
                [],
            ),
            dim=-3,
        )  # [n_total_perms, max_n_atoms, 3] or [batch, n_total_perms, max_n_atoms, 3]
        assert pred_coord.shape[-3:] == (n_total_perms, max_n_atoms, 3)

        batch_shape = pred_coord.shape[:-3]
        assert len(batch_shape) in [0, 1]

        # Expand permuted coords to match batch dimension if needed
        permuted_coords = permuted.coords
        permuted_mask = permuted.coord_mask
        if len(batch_shape) == 1:
            batch_size = pred_coord.size(0)
            permuted_coords = expand_at_dim(permuted_coords, dim=0, n=batch_size)
            permuted_mask = expand_at_dim(permuted_mask, dim=0, n=batch_size)

        # Compute RMSD for all permutations at once
        with torch.amp.autocast("cuda", enabled=False):
            all_rmsds = rmsd(
                pred_pose=pred_coord.to(torch.float32),
                true_pose=permuted_coords.to(torch.float32),
                mask=permuted_mask,
                eps=eps,
                reduce=False,
            )  # [n_total_perms] or [batch, n_total_perms]
            assert all_rmsds.size() == batch_shape + (n_total_perms,)

        # Find best permutation for each residue
        best_permutation_list: list[torch.Tensor] = []
        is_permuted_list: list[torch.Tensor] = []
        optimized_rmsd_list: list[torch.Tensor] = []
        original_rmsd_list: list[torch.Tensor] = []

        perm_idx = 0
        for residue, n_perms in zip(residues, per_residue_n_perms):
            # Extract RMSDs for this residue's permutations
            res_rmsds = all_rmsds[
                ..., perm_idx : perm_idx + n_perms
            ]  # [batch?, n_perms]

            # Find best permutation
            best_rmsd, best_j = torch.min(res_rmsds, dim=-1)  # [batch?]
            best_perm = residue.permutations[best_j]  # [batch?, n_atoms]

            best_permutation_list.append(best_perm)
            is_permuted_list.append(best_j > 0)  # Identity is always first
            optimized_rmsd_list.append(best_rmsd)
            original_rmsd_list.append(res_rmsds[..., 0])

            if run_checker:
                n_atoms = residue.n_atoms
                assert res_rmsds.size() == batch_shape + (n_perms,)
                assert best_rmsd.size() == batch_shape
                assert best_j.size() == batch_shape
                assert best_perm.size() == batch_shape + (n_atoms,)
                Checker.are_permutations(best_perm, dim=-1)

                def _check_identity(j_val: torch.Tensor, perm: torch.Tensor) -> None:
                    if j_val > 0:
                        Checker.not_contain_identity(perm)
                    else:
                        Checker.contains_identity(perm)

                if best_j.dim() == 0:
                    _check_identity(best_j, best_perm)
                else:
                    for j_val, perm_j in zip(best_j, best_perm):
                        _check_identity(j_val, perm_j)

            perm_idx += n_perms

        return (
            best_permutation_list,
            is_permuted_list,
            optimized_rmsd_list,
            original_rmsd_list,
        )

    @typecheck
    def __call__(
        self,
        pred_coord: Float[Tensor, "*batch n 3"],
        true_coord: Float[Tensor, "n 3"],
        true_coord_mask: Bool[Tensor, "n"],
        ref_space_uid: Int[Tensor, "n"],
        atom_perm_list: list[list[int]],
        alignment_mask: Optional[Bool[Tensor, "n"]] = None,
        verbose: bool = False,
    ) -> AtomPermutationResult:
        """Find optimal atom permutation to minimize RMSD.

        Args:
            pred_coord: Predicted atom coordinates, optionally batched.
            true_coord: Ground truth atom coordinates.
            true_coord_mask: Mask indicating resolved atoms.
            ref_space_uid: Unique residue ID for each atom.
            atom_perm_list: Per-atom permutation indices. len == n_atoms.
            alignment_mask: Optional mask for alignment (defaults to true_coord_mask).
            verbose: If True, log additional info.

        Returns:
            AtomPermutationResult containing optimal permutation and statistics.
        """
        device = pred_coord.device
        batch_shape = pred_coord.shape[:-2]
        n_atoms = pred_coord.size(-2)

        # Initialize permutation as identity
        permutation = self.get_identity_permutation(batch_shape, n_atoms, device)

        # Collect residues that require permutations
        symmetric_residues = collect_residues_with_symmetric_atoms(
            coord=true_coord,
            coord_mask=true_coord_mask,
            ref_space_uid=ref_space_uid,
            atom_perm_list=atom_perm_list,
            run_checker=self.run_checker,
        )

        # Build initial stats (may be updated below)
        stats = PermutationStats(
            n_residues=symmetric_residues.total_residue_count,
            n_residues_with_symmetry=len(symmetric_residues),
            n_residues_permuted=0.0,
            has_residue_permuted=0.0,
        )

        # If no residues contain symmetry, return identity
        if symmetric_residues.is_empty():
            logger.info(
                "No atom permutation is needed. Return the identity permutation."
            )
            return AtomPermutationResult(permutation=permutation, stats=stats)

        # Build mask for non-permutable atoms (1 = cannot be permuted)
        no_permute_atom_mask = torch.ones_like(true_coord_mask)
        for residue in symmetric_residues:
            has_non_identity = (residue.permutations != residue.permutations[0]).sum(
                dim=0
            ) > 0
            no_permute_atom_mask[residue.start : residue.end] = ~has_non_identity

        # Compute effective alignment mask
        if alignment_mask is None:
            effective_alignment_mask = true_coord_mask
        else:
            effective_alignment_mask = true_coord_mask & alignment_mask
        if self.global_align_wo_symmetric_atom:
            effective_alignment_mask = no_permute_atom_mask & effective_alignment_mask

        # Need at least 3 atoms for alignment
        if effective_alignment_mask.sum().item() < 3:
            logger.info(
                "No atom permutation is needed. Return the identity permutation."
            )
            return AtomPermutationResult(permutation=permutation, stats=stats)

        # Align predictions to ground truth
        _, transformed_pred_coord = self.global_align_pred_to_true(
            pred_coord=pred_coord,
            true_coord=true_coord,
            true_coord_mask=effective_alignment_mask,
            eps=self.eps,
        )

        # Compute baseline RMSD (before permutation)
        aligned_rmsd, _ = self.global_align_pred_to_true(
            pred_coord=pred_coord,
            true_coord=true_coord,
            true_coord_mask=true_coord_mask,
            eps=self.eps,
        )
        unpermuted_rmsd = aligned_rmsd.mean().item()

        # Group residues by atom count for efficient batched optimization
        res_atom_cutoffs = [15, 30, 50, 100, 100000]
        grouped_indices: dict[int, list[int]] = {}
        for i, residue in enumerate(symmetric_residues):
            for cutoff in res_atom_cutoffs:
                if residue.n_atoms <= cutoff:
                    grouped_indices.setdefault(cutoff, []).append(i)
                    break

        residue_position_list: list[tuple[int, int]] = []
        residue_best_permutation_list: list[Tensor] = []
        residue_is_permuted_list: list[Tensor] = []

        residue_list = list(symmetric_residues.residues)
        for cutoff, residue_group in grouped_indices.items():
            if verbose:
                logger.info(f"{len(residue_group)} residues have <={cutoff} atoms.")

            group_residues = [residue_list[i] for i in residue_group]
            (
                per_res_best_permutation,
                per_res_is_permuted,
                _,  # optimized_rmsd (unused)
                _,  # original_rmsd (unused)
            ) = self._optimize_per_residue_permutation_by_rmsd(
                pred_coords_list=[
                    transformed_pred_coord[..., r.start : r.end, :]
                    for r in group_residues
                ],
                residues=group_residues,
                eps=self.eps,
                run_checker=self.run_checker,
            )
            residue_position_list.extend((r.start, r.end) for r in group_residues)
            residue_best_permutation_list.extend(per_res_best_permutation)
            residue_is_permuted_list.extend(per_res_is_permuted)

        # Apply best permutations to full permutation tensor
        indices_list = [
            torch.arange(start, end, device=device)
            for start, end in residue_position_list
        ]
        residue_atom_indices = torch.cat(indices_list, dim=-1)
        residue_best_permutation = torch.cat(
            [
                idx[perm]
                for idx, perm in zip(indices_list, residue_best_permutation_list)
            ],
            dim=-1,
        )
        permutation[..., residue_atom_indices] = residue_best_permutation

        # Compute final statistics
        is_res_permuted = torch.stack(residue_is_permuted_list, dim=-1).float()
        stats = PermutationStats(
            n_residues=symmetric_residues.total_residue_count,
            n_residues_with_symmetry=len(symmetric_residues),
            n_residues_permuted=is_res_permuted.sum(dim=-1).mean().item(),
            has_residue_permuted=(is_res_permuted.sum(dim=-1) > 0)
            .float()
            .mean()
            .item(),
            unpermuted_rmsd=unpermuted_rmsd,
        )

        return AtomPermutationResult(permutation=permutation, stats=stats)


@typecheck
def correct_symmetric_atoms(
    pred_coord: Float[Tensor, "*batch n 3"],
    true_coord: Float[Tensor, "n 3"],
    true_coord_mask: Bool[Tensor, "n"],
    ref_space_uid: Int[Tensor, "n"],
    atom_perm_list: list[list[int]],
    permute_label: bool = True,
    alignment_mask: Optional[Bool[Tensor, "n"]] = None,
    verbose: bool = False,
    run_checker: bool = False,
    eps: float = 1e-8,
    global_align_wo_symmetric_atom: bool = False,
) -> PermutationResult:
    """Correct symmetric atoms by finding optimal permutation.

    Args:
        pred_coord: Predicted atom coordinates, optionally batched.
        true_coord: Ground truth atom coordinates.
        true_coord_mask: Mask indicating resolved atoms.
        ref_space_uid: Unique residue ID for each atom.
        atom_perm_list: Per-atom permutation indices.
        permute_label: If True, permute ground truth; if False, permute prediction.
        alignment_mask: Optional mask for alignment.
        verbose: If True, log additional info.
        run_checker: If True, run additional validity checks.
        eps: Small number for numerical stability.
        global_align_wo_symmetric_atom: If True, exclude symmetric atoms from alignment.

    Returns:
        PermutedLabel if permute_label=True, PermutedPrediction otherwise.
    """
    # Compute effective alignment mask
    if alignment_mask is not None:
        effective_alignment_mask = true_coord_mask & alignment_mask
    else:
        effective_alignment_mask = true_coord_mask

    with torch.no_grad():
        atom_perm = AtomPermutation(
            run_checker=run_checker,
            eps=eps,
            global_align_wo_symmetric_atom=global_align_wo_symmetric_atom,
        )
        result = atom_perm(
            pred_coord=pred_coord,
            true_coord=true_coord,
            true_coord_mask=true_coord_mask,
            ref_space_uid=ref_space_uid,
            atom_perm_list=atom_perm_list,
            alignment_mask=effective_alignment_mask,
            verbose=verbose,
        )

    permutation = result.permutation
    stats = result.stats

    # Compute permuted RMSD if we have baseline
    if stats.unpermuted_rmsd is not None:
        permuted_rmsd, _ = AtomPermutation.global_align_pred_to_true(
            pred_coord,
            true_coord[permutation],
            true_coord_mask[permutation],
            eps=eps,
        )
        stats = PermutationStats(
            n_residues=stats.n_residues,
            n_residues_with_symmetry=stats.n_residues_with_symmetry,
            n_residues_permuted=stats.n_residues_permuted,
            has_residue_permuted=stats.has_residue_permuted,
            unpermuted_rmsd=stats.unpermuted_rmsd,
            permuted_rmsd=permuted_rmsd.mean().item(),
        )

    if permute_label:
        return PermutedLabel(
            coordinate=true_coord[permutation],
            coordinate_mask=true_coord_mask[permutation],
            permutation=permutation,
            stats=stats,
        )
    else:
        # Compute inverse permutation to apply to predictions
        if pred_coord.dim() == 2:
            inv_permutation = torch.argsort(permutation)
            pred_coord_permuted = pred_coord[inv_permutation]
        else:
            inv_permutation = torch.argsort(permutation, dim=1)
            inv_permutation_expanded = expand_at_dim(inv_permutation, dim=-1, n=3)
            pred_coord_permuted = pred_coord.gather(1, inv_permutation_expanded)

        return PermutedPrediction(
            coordinate=pred_coord_permuted,
            permutation=inv_permutation,
            stats=stats,
        )


@typecheck
def run(
    pred_coord: Float[Tensor, "n_samples n 3"],
    true_coord: Float[Tensor, "n 3"],
    true_coord_mask: Bool[Tensor, "n"],
    ref_space_uid: Int[Tensor, "n"],
    atom_perm_list: list[list[int]],
    permute_label: bool = True,
    alignment_mask: Optional[Bool[Tensor, "n"]] = None,
    error_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    pdb_id: Optional[str] = None,
    global_align_wo_symmetric_atom: bool = False,
) -> Optional[PermutationResult]:
    """Apply permutation to correct symmetric atoms, with error handling.

    Args:
        pred_coord: Predicted atom coordinates, optionally batched.
        true_coord: Ground truth atom coordinates.
        true_coord_mask: Mask indicating resolved atoms.
        ref_space_uid: Unique residue ID for each atom.
        atom_perm_list: Per-atom permutation indices.
        permute_label: If True, permute ground truth; if False, permute prediction.
        alignment_mask: Optional mask for alignment.
        error_dir: Directory to save error information on failure.
        dataset_name: Dataset name for error logging.
        pdb_id: PDB ID for error logging.
        global_align_wo_symmetric_atom: If True, exclude symmetric atoms from alignment.

    Returns:
        PermutationResult on success, None on failure.
    """
    try:
        return correct_symmetric_atoms(
            pred_coord=pred_coord,
            true_coord=true_coord,
            true_coord_mask=true_coord_mask,
            ref_space_uid=ref_space_uid,
            atom_perm_list=atom_perm_list,
            permute_label=permute_label,
            alignment_mask=alignment_mask,
            global_align_wo_symmetric_atom=global_align_wo_symmetric_atom,
        )

    except Exception as e:
        error_message = str(e)
        if dataset_name:
            logger.warning(f"dataset: {dataset_name}")
        if pdb_id:
            logger.warning(f"pdb id: {pdb_id}")
        logger.warning(error_message)
        save_permutation_error(
            data={
                "error_message": error_message,
                "pred_coord": pred_coord,
                "true_coord": true_coord,
                "true_coord_mask": true_coord_mask,
                "ref_space_uid": ref_space_uid,
                "atom_perm_list": atom_perm_list,
                "permute_label": permute_label,
                "alignment_mask": alignment_mask,
                "dataset_name": dataset_name,
                "pdb_id": pdb_id,
            },
            error_dir=error_dir,
        )
        return None


@typecheck
def run_batched(
    pred_coord: Float[Tensor, "batch n_samples n 3"],
    true_coord: Float[Tensor, "batch n 3"],
    true_coord_mask: Bool[Tensor, "batch n"],
    ref_space_uid: Int[Tensor, "batch n"],
    atom_perm_list: list[list[list[int]]],
    pdb_id: list[str],
    permute_label: bool = False,
    alignment_mask: Optional[Bool[Tensor, "batch n"]] = None,
    error_dir: Optional[str] = None,
    global_align_wo_symmetric_atom: bool = False,
) -> Optional[PermutationResult]:
    """Apply permutation to correct symmetric atoms for batched input.

    Loops over the batch dimension and calls `run` for each sample independently.

    Args:
        pred_coord: Predicted coordinates. Shape: [batch, n_samples, n, 3].
        true_coord: Ground truth coordinates. Shape: [batch, n, 3].
        true_coord_mask: Mask for resolved atoms. Shape: [batch, n].
        ref_space_uid: Unique residue ID per atom. Shape: [batch, n].
        atom_perm_list: Per-atom permutation indices. Length: batch.
        pdb_id: List of pdb ids in the batch
        permute_label: If True, permute ground truth; if False, permute prediction.
        alignment_mask: Optional mask for alignment. Shape: [batch, n].
        error_dir: Directory to save error information on failure.
        global_align_wo_symmetric_atom: If True, exclude symmetric atoms from alignment.

    Returns:
        PermutedLabel if permute_label=True, PermutedPrediction otherwise.
        Tensors are stacked along batch dimension, stats is a list.
        Returns None if any sample fails.
    """
    batch_size = pred_coord.size(0)

    coords_list: list[Tensor] = []
    perms_list: list[Tensor] = []
    masks_list: list[Tensor] = []
    stats_list: list[PermutationStats] = []

    for i in range(batch_size):
        align_mask_i = alignment_mask[i] if alignment_mask is not None else None

        result = run(
            pred_coord=pred_coord[i],
            true_coord=true_coord[i],
            true_coord_mask=true_coord_mask[i],
            ref_space_uid=ref_space_uid[i],
            atom_perm_list=atom_perm_list[i],
            pdb_id=pdb_id[i],
            permute_label=permute_label,
            alignment_mask=align_mask_i,
            error_dir=error_dir,
            global_align_wo_symmetric_atom=global_align_wo_symmetric_atom,
        )

        if result is None:
            return None

        coords_list.append(result.coordinate)
        perms_list.append(result.permutation)
        stats_list.append(result.stats)
        if isinstance(result, PermutedLabel):
            masks_list.append(result.coordinate_mask)

    permutation = torch.stack(perms_list, dim=0)

    if permute_label:
        return PermutedLabel(
            coordinate=torch.stack(coords_list, dim=0),
            coordinate_mask=torch.stack(masks_list, dim=0),
            permutation=permutation,
            stats=stats_list,
        )
    else:
        return PermutedPrediction(
            coordinate=torch.stack(coords_list, dim=0),
            permutation=permutation,
            stats=stats_list,
        )
