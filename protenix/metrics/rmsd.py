from typing import Literal, Optional, overload

import numpy as np
import torch
from biotite.structure import AtomArray
from einops import rearrange
from torch import Tensor

from protenix.utils.typing import Bool, Float, typecheck


@typecheck
def rmsd(
    pred_pose: Float[Tensor, "*batch n 3"],
    true_pose: Float[Tensor, "*batch n 3"],
    mask: Optional[Bool[Tensor, "*batch n"]] = None,
    eps: float = 0.0,
    reduce: bool = True,
) -> Float[Tensor, ""] | Float[Tensor, "*batch"]:
    """Compute RMSD between two poses with the same shape.

    Args:
        pred_pose: Predicted pose coordinates.
        true_pose: Ground truth pose coordinates.
        mask: Mask indicating which atoms to include in computation.
        eps: Small tolerance to avoid floating point issues in sqrt.
        reduce: If True, return mean RMSD over batches; else return per-batch RMSD.

    Returns:
        RMSD value(s) - scalar if reduce=True, otherwise tensor of shape [*batch].
    """
    if mask is None:
        mask = torch.ones(true_pose.shape[:-1], device=true_pose.device)

    err2 = (torch.square(pred_pose - true_pose).sum(dim=-1) * mask).sum(
        dim=-1
    ) / mask.sum(dim=-1)
    rmsd = err2.add(eps).sqrt()
    if reduce:
        rmsd = rmsd.mean()
    return rmsd


@typecheck
def align_pred_to_true(
    pred_pose: Float[Tensor, "*batch n 3"],
    true_pose: Float[Tensor, "*batch n 3"],
    atom_mask: Optional[Bool[Tensor, "*batch n"]] = None,
    weight: Optional[Float[Tensor, "*batch n"]] = None,
    allowing_reflection: bool = False,
) -> tuple[
    Float[Tensor, "*batch n 3"],
    Float[Tensor, "*batch 3 3"],
    Float[Tensor, "*batch 1 3"],
]:
    """Find optimal transformation, rotation (and reflection) of two poses.

    Uses Kabsch algorithm (SVD-based) to find the optimal rotation matrix.

    Args:
        pred_pose: The pose to perform transformation on.
        true_pose: The target pose to align pred_pose to.
        atom_mask: A mask for atoms.
        weight: A weight vector to be applied.
        allowing_reflection: Whether to allow reflection when finding optimal alignment.

    Returns:
        Tuple of (aligned_pose, rotation_matrix, translation_vector).
    """
    if atom_mask is not None:
        pred_pose = pred_pose * atom_mask[..., None]
        true_pose = true_pose * atom_mask[..., None]
    else:
        atom_mask = torch.ones(*pred_pose.shape[:-1], device=pred_pose.device)

    if weight is None:
        weight = atom_mask.float()
    else:
        weight = weight * atom_mask

    # Compute weighted centroids: [*batch, 1, 3]
    weighted_n_atoms = rearrange(weight.sum(dim=-1), "... -> ... 1 1")
    weight = rearrange(weight, "... n -> ... n 1")

    pred_pose_centroid = (pred_pose * weight).sum(
        dim=-2, keepdim=True
    ) / weighted_n_atoms
    true_pose_centroid = (true_pose * weight).sum(
        dim=-2, keepdim=True
    ) / weighted_n_atoms

    # Center the poses
    pred_pose_centered = pred_pose - pred_pose_centroid
    true_pose_centered = true_pose - true_pose_centroid

    # Compute cross-covariance matrix H: [*batch, 3, 3]
    # H = P^T @ Q where P is weighted pred, Q is masked true
    h_mat = torch.matmul(
        rearrange(pred_pose_centered * weight, "... n c -> ... c n"),
        true_pose_centered * rearrange(atom_mask, "... n -> ... n 1"),
    )

    # SVD: H = U @ S @ V^T
    u, _s, v = torch.svd(h_mat)
    u_t = rearrange(u, "... i j -> ... j i")

    # Compute optimal rotation
    if not allowing_reflection:
        det = torch.linalg.det(torch.matmul(v, u_t))
        diagonal = torch.stack(
            [torch.ones_like(det), torch.ones_like(det), det], dim=-1
        )
        rot = torch.matmul(v, torch.matmul(torch.diag_embed(diagonal), u_t))
    else:
        rot = torch.matmul(v, u_t)

    # Compute translation: [*batch, 1, 3]
    rot_t = rearrange(rot, "... i j -> ... j i")
    translate = true_pose_centroid - torch.matmul(pred_pose_centroid, rot_t)

    # Apply transformation
    pred_pose_aligned = torch.matmul(pred_pose_centered, rot_t) + true_pose_centroid

    return pred_pose_aligned, rot, translate


@typecheck
def partially_aligned_rmsd(
    pred_pose: Float[Tensor, "*batch n 3"],
    true_pose: Float[Tensor, "*batch n 3"],
    align_mask: Bool[Tensor, "*batch n"],
    atom_mask: Bool[Tensor, "*batch n"],
    weight: Optional[Float[Tensor, "*batch n"]] = None,
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
) -> tuple[
    Float[Tensor, ""] | Float[Tensor, "*batch"],
    Float[Tensor, ""] | Float[Tensor, "*batch"],
    Float[Tensor, "*batch n 3"],
    Float[Tensor, "*batch 3 3"],
    Float[Tensor, "*batch 1 3"],
]:
    """RMSD when aligning parts of the complex, ignoring permutation symmetry.

    Args:
        pred_pose: Native predicted pose.
        true_pose: Ground truth pose.
        align_mask: Mask for which coordinates to align.
        atom_mask: Mask for which coordinates to compute loss.
        weight: Weight tensor for each atom in alignment.
        eps: Tolerance to avoid floating point issues in sqrt.
        reduce: If True, return mean RMSD; else per-batch RMSD.
        allowing_reflection: Whether to allow reflection in alignment.

    Returns:
        Tuple of (aligned_rmsd, unaligned_rmsd, transformed_pose, rotation, translation).
    """
    _, rot, translate = align_pred_to_true(
        pred_pose,
        true_pose,
        atom_mask=atom_mask & align_mask,
        weight=weight,
        allowing_reflection=allowing_reflection,
    )
    rot_t = rearrange(rot, "... i j -> ... j i")
    transformed_pose = torch.matmul(pred_pose, rot_t) + translate

    err_atom = torch.square(transformed_pose - true_pose).sum(dim=-1) * atom_mask
    aligned_mask = atom_mask & align_mask
    unaligned_mask = atom_mask & ~align_mask

    aligned_part_err_square = (err_atom * aligned_mask).sum(dim=-1) / aligned_mask.sum(
        dim=-1
    )
    unaligned_part_err_square = (err_atom * unaligned_mask).sum(
        dim=-1
    ) / unaligned_mask.sum(dim=-1)

    aligned_part_rmsd = aligned_part_err_square.add(eps).sqrt()
    unaligned_part_rmsd = unaligned_part_err_square.add(eps).sqrt()

    if reduce:
        aligned_part_rmsd = aligned_part_rmsd.mean()
        unaligned_part_rmsd = unaligned_part_rmsd.mean()

    return aligned_part_rmsd, unaligned_part_rmsd, transformed_pose, rot, translate


@typecheck
def self_aligned_rmsd(
    pred_pose: Float[Tensor, "*batch n 3"],
    true_pose: Float[Tensor, "*batch n 3"],
    atom_mask: Bool[Tensor, "*batch n"],
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
) -> tuple[
    Float[Tensor, ""] | Float[Tensor, "*batch"],
    Float[Tensor, "*batch n 3"],
    Float[Tensor, "*batch 3 3"],
    Float[Tensor, "*batch 1 3"],
]:
    """RMSD when aligning one molecule with ground truth.

    Args:
        pred_pose: Native predicted pose.
        true_pose: Ground truth pose.
        atom_mask: Mask for which coordinates to compute loss.
        eps: Tolerance to avoid floating point issues in sqrt.
        reduce: If True, return mean RMSD; else per-batch RMSD.
        allowing_reflection: Whether to allow reflection in alignment.

    Returns:
        Tuple of (aligned_rmsd, transformed_pose, rotation, translation).
    """
    aligned_rmsd, _, transformed_pred_pose, rot, trans = partially_aligned_rmsd(
        pred_pose=pred_pose,
        true_pose=true_pose,
        align_mask=atom_mask,
        atom_mask=atom_mask,
        eps=eps,
        reduce=reduce,
        allowing_reflection=allowing_reflection,
    )
    return aligned_rmsd, transformed_pred_pose, rot, trans


@overload
def weighted_rigid_align(
    x: Float[Tensor, "*batch n 3"],
    x_target: Float[Tensor, "*batch n 3"],
    atom_weight: Float[Tensor, "*batch n"] | Float[Tensor, "n"],
    mask: Bool[Tensor, "*batch n"],
    stop_gradient: bool = True,
    return_transform: Literal[False] = False,
) -> Float[Tensor, "*batch n 3"]: ...


@overload
def weighted_rigid_align(
    x: Float[Tensor, "*batch n 3"],
    x_target: Float[Tensor, "*batch n 3"],
    atom_weight: Float[Tensor, "*batch n"] | Float[Tensor, "n"],
    mask: Bool[Tensor, "*batch n"],
    stop_gradient: bool,
    return_transform: Literal[True],
) -> tuple[
    Float[Tensor, "*batch n 3"],
    Float[Tensor, "*batch 3 3"],
    Float[Tensor, "*batch 1 3"],
]: ...


@typecheck
def weighted_rigid_align(
    x: Float[Tensor, "*batch n 3"],
    x_target: Float[Tensor, "*batch n 3"],
    atom_weight: Float[Tensor, "*batch n"] | Float[Tensor, "n"],
    mask: Bool[Tensor, "*batch n"],
    stop_gradient: bool = True,
    return_transform: bool = False,
) -> (
    Float[Tensor, "*batch n 3"]
    | tuple[
        Float[Tensor, "*batch n 3"],
        Float[Tensor, "*batch 3 3"],
        Float[Tensor, "*batch 1 3"],
    ]
):
    """Implements Algorithm 28 in AF3. Wraps `align_pred_to_true`.

    Args:
        x: Input coordinates to be transformed.
        x_target: Target coordinates to align x to.
        atom_weight: Weights for each atom.
        mask: Mask for the atoms.
        stop_gradient: If True, detach output and run with no_grad.
        return_transform: If True, also return rotation and translation.

    Returns:
        If return_transform is False: aligned coordinates only.
        If return_transform is True: tuple of (aligned_coords, rotation, translation).
    """
    if stop_gradient:
        with torch.no_grad():
            x_aligned, rot, trans = align_pred_to_true(
                pred_pose=x,
                true_pose=x_target,
                atom_mask=mask,
                weight=atom_weight,
                allowing_reflection=False,
            )
            x_aligned = x_aligned.detach()
            rot = rot.detach()
            trans = trans.detach()
    else:
        x_aligned, rot, trans = align_pred_to_true(
            pred_pose=x,
            true_pose=x_target,
            atom_mask=mask,
            weight=atom_weight,
            allowing_reflection=False,
        )

    if return_transform:
        return x_aligned, rot, trans
    return x_aligned


def compute_per_atom_rmsf(
    structures: list[AtomArray],
) -> np.ndarray:
    """Compute per-atom RMSF across an ensemble of structures.

    All structures must have the same atoms in the same order.

    Args:
        structures: List of AtomArrays, all with identical atom ordering

    Returns:
        Array of shape (n_atoms,) containing RMSF for each atom in Angstroms
    """
    all_coords = np.stack(
        [s.coord for s in structures], axis=0
    )  # [n_structures, n_atoms, 3]

    mean_coords = np.mean(all_coords, axis=0)  # [n_atoms, 3]

    squared_deviations = np.sum(
        (all_coords - mean_coords) ** 2, axis=2
    )  # [n_structures, n_atoms]

    rmsf = np.sqrt(np.mean(squared_deviations, axis=0))  # [n_atoms]

    return rmsf


def compute_per_residue_rmsf(
    structure: AtomArray,
    per_atom_rmsf: np.ndarray,
) -> dict[tuple[str, int], float]:
    """Aggregate per-atom RMSF to per-residue RMSF.

    Args:
        structure: Reference AtomArray
        per_atom_rmsf: Array of per-atom RMSF values

    Returns:
        Dict mapping (chain_id, res_id) to mean RMSF over atoms in that residue
    """
    residue_rmsf = {}

    chain_res_pairs = list(zip(structure.chain_id, structure.res_id))
    unique_pairs = sorted(set(chain_res_pairs), key=lambda x: (x[0], x[1]))

    for chain_id, res_id in unique_pairs:
        mask = (structure.chain_id == chain_id) & (structure.res_id == res_id)
        atom_rmsf_values = per_atom_rmsf[mask]
        if len(atom_rmsf_values) > 0:
            residue_rmsf[(chain_id, res_id)] = float(np.mean(atom_rmsf_values))

    return residue_rmsf
