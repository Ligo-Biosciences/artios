# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.spatial.transform import Rotation


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
