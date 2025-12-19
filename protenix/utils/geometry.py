import numpy as np
import torch
from einops import repeat
from scipy.spatial.transform import Rotation
from torch import Tensor

from protenix.utils.typing import Bool, Float, typecheck

# =============================================================================
# Torch geometry utilities
# =============================================================================


@typecheck
def calc_centroid(
    coords: Float[Tensor, "*batch n_atom xyz"],
    mask: Bool[Tensor, "*batch n_atom"] | None = None,
    eps: float = 1e-12,
) -> Float[Tensor, "... xyz"]:
    """Calculate the centroid (center of mass) of coordinates.

    Args:
        coords: Input coordinates.
        mask: Optional mask for valid atoms. If None, all atoms are considered valid.
        eps: Small epsilon to avoid division by zero.

    Returns:
        The centroid of the coordinates.
    """
    if mask is None:
        return coords.mean(dim=-2)
    else:
        masked_coords = coords * mask.unsqueeze(dim=-1)
        centroid = masked_coords.sum(dim=-2) / (mask.sum(dim=-1, keepdim=True) + eps)
        return centroid


@typecheck
def centre_random_augmentation(
    x_input_coords: Float[Tensor, "*batch n_atom xyz"],
    n_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: Bool[Tensor, "*batch n_atom"] | None = None,
    eps: float = 1e-12,
) -> Float[Tensor, "... n_sample n_atom xyz"]:
    """Implements Algorithm 19 in AF3.

    Args:
        x_input_coords: Input coords.
        n_sample: The total number of augmentation. Defaults to 1.
        s_trans: Scale factor of trans. Defaults to 1.0.
        centre_only: If set true, will only perform centering without applying
            random translation and rotation.
        mask: Masking for the coords.
        eps: Small number used for masked mean.

    Returns:
        The augmentation version of input coords.
    """
    n_atom = x_input_coords.size(-2)
    device = x_input_coords.device

    # Create all-True mask if not provided
    if mask is None:
        mask = torch.ones_like(x_input_coords[..., 0], dtype=torch.bool)

    # Move to origin [..., n_atom, 3]
    centroid = calc_centroid(x_input_coords, mask=mask, eps=eps)
    x_input_coords = x_input_coords - centroid.unsqueeze(dim=-2)

    # Expand to [..., n_sample, n_atom, 3]
    x_input_coords = repeat(
        x_input_coords, "... n_atom xyz -> ... n_sample n_atom xyz", n_sample=n_sample
    )

    if centre_only:
        x_input_coords = x_input_coords * mask[..., None, :, None]
        return x_input_coords

    # n_augment = batch_size * n_sample
    n_augment = torch.numel(x_input_coords[..., 0, 0])

    # Generate n_augment (rot, trans) pairs
    batch_size_shape = x_input_coords.shape[:-3]
    rot_matrix_random = (
        uniform_random_rotation(n_sample=n_augment)
        .to(device)
        .reshape(*batch_size_shape, n_sample, 3, 3)
    ).detach()  # [..., n_sample, 3, 3]
    trans_random = s_trans * torch.randn(size=(*batch_size_shape, n_sample, 3)).to(
        device
    )  # [..., n_sample, 3]
    rot_expanded = repeat(
        rot_matrix_random, "... n_sample u v -> ... n_sample n_atom u v", n_atom=n_atom
    )
    x_augment_coords = (
        rot_vec_mul(r=rot_expanded, t=x_input_coords) + trans_random[..., None, :]
    )  # [..., n_sample, n_atom, 3]

    # Apply the mask
    x_augment_coords = x_augment_coords * mask[..., None, :, None]
    return x_augment_coords


# Note: Rotation.random is not supported by torch.compile()
@typecheck
def uniform_random_rotation(n_sample: int = 1) -> Float[Tensor, "n_sample u v"]:
    """Generate random rotation matrices with scipy.spatial.transform.Rotation.

    Args:
        n_sample: The total number of rotation matrices to generate.

    Returns:
        n_sample rotation matrices [n_sample, 3, 3].
    """
    rotation = Rotation.random(num=n_sample)
    rot_matrix = torch.from_numpy(rotation.as_matrix()).float()
    return rot_matrix


@typecheck
def rot_vec_mul(
    r: Float[Tensor, "*batch u v"], t: Float[Tensor, "*batch xyz"]
) -> Float[Tensor, "*batch xyz"]:
    """Apply rotation matrix to vector.

    Written out by hand to avoid AMP downcasting.

    Args:
        r: The rotation matrices.
        t: The coordinate tensors.

    Returns:
        The rotated coordinates.
    """
    x, y, z = torch.unbind(input=t, dim=-1)
    return torch.stack(
        tensors=[
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


# =============================================================================
# Numpy geometry utilities
# =============================================================================


def angle_3p(a, b, c):
    """
    Calculate the angle between three points in a 2D space.

    Args:
        a (list or array-like): The coordinates of the first point.
        b (list or array-like): The coordinates of the second point.
        c (list or array-like): The coordinates of the third point.

    Returns:
        float: The angle in degrees (0, 180) between the vectors
               from point a to point b and point b to point c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = b - a
    bc = c - b

    dot_product = np.dot(ab, bc)

    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)

    cos_theta = np.clip(dot_product / (norm_ab * norm_bc + 1e-4), -1, 1)
    theta_radians = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta_radians)
    return theta_degrees


def random_transform(
    points: np.ndarray,
    max_translation: float = 1.0,
    apply_augmentation: bool = False,
    centralize: bool = True,
    mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Randomly transform a set of 3D points.

    Args:
        points: The points to be transformed, shape=(N, 3)
        max_translation: The maximum translation value.
        apply_augmentation: Whether to apply random rotation/translation
        centralize: Whether to center the points
        mask: Boolean mask for valid atoms, shape=(N,). If provided, only
            masked atoms are used for computing the center of mass, and
            unmasked coordinates are zeroed after transformation.
        eps: Small value to avoid division by zero

    Returns:
        The transformed points.
    """
    # Expand mask to (N, 1) for broadcasting if provided
    mask_expanded = None
    if mask is not None:
        mask_expanded = mask[:, np.newaxis].astype(points.dtype)

    if centralize:
        if mask_expanded is None:
            center = points.mean(axis=0)
        else:
            center = (points * mask_expanded).sum(axis=0) / (mask_expanded.sum() + eps)
        points = points - center

    if apply_augmentation:
        translation = np.random.uniform(-max_translation, max_translation, size=3)
        R = Rotation.random().as_matrix()
        points = np.dot(points + translation, R.T)

    if mask_expanded is not None:
        points = points * mask_expanded

    return points


def compute_virtual_cb(
    n_coord: np.ndarray,
    ca_coord: np.ndarray,
    c_coord: np.ndarray,
) -> np.ndarray:
    """Compute virtual Cβ position using ideal geometry from ProteinMPNN.

    Uses idealized bond lengths and angles to place Cβ in the correct
    tetrahedral position relative to the N-CA-C backbone.

    Args:
        n_coord: N atom coordinates [..., 3]
        ca_coord: CA atom coordinates [..., 3]
        c_coord: C atom coordinates [..., 3]

    Returns:
        numpy.ndarray: Virtual Cβ coordinates

    Reference:
        ProteinMPNN implementation of virtual Cβ placement
    """
    n_ca = ca_coord - n_coord
    ca_c = c_coord - ca_coord
    cross = np.cross(n_ca, ca_c)
    cb_coord = ca_coord + 0.568 * n_ca - 0.54068 * ca_c - 0.5827 * cross
    return cb_coord
