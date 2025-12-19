"""High-level API for pocket detection from Pocketeer.

Source code for Pocketeer: https://github.com/cch1999/pocketeer/tree/main

The polarity detection algorithm label_polarity is modified to use RASA.
"""

import logging

import biotite.structure as struc  # type: ignore
import numpy as np
from pocketeer.core import (
    Pocket,
    cluster_spheres,
    compute_alpha_spheres,
    create_pocket,
    filter_surface_spheres,
    # label_polarity,
)
from pocketeer.core.types import AlphaSphere
from pocketeer.utils.constants import MIN_ATOMS_FOR_TESSELLATION
from pocketeer.utils.exceptions import ValidationError
from scipy.spatial import KDTree

logger = logging.getLogger("pocketeer.api")


def label_polarity(
    spheres: list[AlphaSphere],
    atomarray: struc.AtomArray,
    polar_probe_radius: float,
    sasa_cutoff: float,
) -> list[AlphaSphere]:
    """Label spheres as buried (apolar) or surface (polar) using SASA.

    Uses solvent accessible surface area (SASA) to determine if atoms
    defining each sphere are buried or exposed to solvent.

    Modifies spheres in place.

    Args:
        spheres: list of alpha-spheres
        atomarray: Biotite AtomArray with structure data
        polar_probe_radius: Probe radius for SASA calculation (Å)

    Returns:
        list[AlphaSphere]: Input spheres with is_buried attribute set
    """
    # Compute SASA for all atoms
    atom_sasa = struc.sasa(atomarray, probe_radius=polar_probe_radius)

    # Label each sphere based on SASA of its defining atoms
    for sphere in spheres:
        # Get SASA values for atoms defining this sphere
        defining_atom_sasa = atom_sasa[sphere.atom_indices]

        # Sphere is buried if all defining atoms have low SASA (< sasa_cutoff Å²)
        sphere.is_buried = bool(np.all(defining_atom_sasa <= sasa_cutoff))

    return spheres


def label_confidence(
    spheres: list[AlphaSphere],
    atomarray: struc.AtomArray,
    plddt_cutoff: float = 70.0,
) -> list[AlphaSphere]:
    """Label spheres as confident or not based on pLDDT scores.

    Uses predicted local distance difference test (pLDDT) scores from the
    b_factor field to determine if atoms defining each sphere are confidently
    predicted.

    Modifies spheres in place.

    Args:
        spheres: list of alpha-spheres
        atomarray: Biotite AtomArray with structure data containing b_factor (pLDDT)
        plddt_cutoff: Minimum pLDDT score for confident prediction. Default: 70.0

    Returns:
        list[AlphaSphere]: Input spheres with is_confident attribute set
    """
    # Get pLDDT scores from b_factor field for all atoms
    atom_plddt = atomarray.b_factor

    # Label each sphere based on pLDDT of its defining atoms
    for sphere in spheres:
        defining_atom_plddt = atom_plddt[sphere.atom_indices]
        sphere.is_confident = bool(np.all(defining_atom_plddt >= plddt_cutoff))

    return spheres


def filter_low_confidence_spheres(
    spheres: list[AlphaSphere],
) -> list[AlphaSphere]:
    """Filter to keep only confident (high pLDDT) spheres for pocket detection.

    Args:
        spheres: list of all alpha-spheres

    Returns:
        List of confident spheres only
    """
    return [s for s in spheres if s.is_confident]


def find_pockets(
    atomarray: struc.AtomArray,
    *,
    r_min: float = 3.0,
    r_max: float = 6.0,
    polar_probe_radius: float = 1.8,
    sasa_cutoff: float = 1.0,
    plddt_cutoff: float | None = None,
    merge_distance: float = 1.75,
    min_spheres: int = 35,
    ignore_hydrogens: bool = True,
    ignore_water: bool = True,
    ignore_hetero: bool = True,
) -> list[Pocket]:
    """Detect binding pockets in a protein structure.

    Main API function for pocket detection using alpha-sphere method.

    The algorithm works as follows:
        1. Compute all alpha-spheres from Delaunay tessellation
        2. Label spheres as buried (interior) or surface and filter to buried ones
        3. Cluster buried spheres into pockets using graph clustering
        4. Create Pocket objects with volume and scoring descriptors

    Args:
        atomarray: Biotite AtomArray with structure data
        r_min: Minimum alpha-sphere radius (Å). Default: 3.0
        r_max: Maximum alpha-sphere radius (Å). Default: 6.0
        polar_probe_radius: Radius to test atom contact for polarity (Å). Default: 1.8
        sasa_cutoff: Solvent accessible surface area cutoff for a buried spheres. Default: 1.0
        plddt_cutoff: AlphaFold pLDDT cutoff for confident spheres. If None, no confidence filtering is applied. Default: None
        merge_distance: Distance threshold for merging nearby sphere clusters (Å). Default: 1.75
        min_spheres: Minimum number of spheres per pocket cluster. Default: 35
        ignore_hydrogens: Ignore hydrogen atoms (recommended). Default: True
        ignore_water: Ignore water molecules (recommended). Default: True
        ignore_hetero: Ignore hetero atoms i.e. ligands (recommended). Default: True

    Returns:
        pockets: list of detected Pocket objects, sorted by score (highest first)

    Raises:
        ValidationError: if atomarray is invalid or empty
        ValueError: if parameters are invalid

    Examples:
        >>> import pocketeer as pt
        >>> atomarray = pt.load_structure("protein.pdb")
        >>> pockets = pt.find_pockets(atomarray)

        >>> # With custom parameters
        >>> pockets = pt.find_pockets(atomarray, r_min=2.5, r_max=7.0, min_spheres=25)
    """
    # Parameter validation
    if r_min <= 0 or r_max <= r_min:
        raise ValueError(f"Invalid radius range: r_min={r_min}, r_max={r_max}")
    if polar_probe_radius <= 0:
        raise ValueError(f"polar_probe_radius must be > 0, got {polar_probe_radius}")

    # Input validation
    if not isinstance(atomarray, struc.AtomArray):
        raise ValidationError(f"Expected AtomArray, got {type(atomarray).__name__}")

    if len(atomarray) == 0:
        raise ValidationError("AtomArray is empty")

    if ignore_hydrogens:
        atomarray = atomarray[atomarray.element != "H"]

    if ignore_water:
        atomarray = atomarray[atomarray.res_name != "HOH"]

    if ignore_hetero:
        atomarray = atomarray[~atomarray.hetero]

    if len(atomarray) < MIN_ATOMS_FOR_TESSELLATION:
        logger.info(f"Not enough atoms ({len(atomarray)}). Returning empty result.")
        return []

    # 1. Compute all alpha-spheres from Delaunay tessellation
    alpha_spheres = compute_alpha_spheres(atomarray.coord, r_min=r_min, r_max=r_max)
    logger.info(f"Computed {len(alpha_spheres)} alpha-spheres")

    # 2. Label spheres as buried or surface and filter
    alpha_spheres = label_polarity(
        alpha_spheres, atomarray, polar_probe_radius, sasa_cutoff
    )
    buried_spheres = filter_surface_spheres(alpha_spheres)
    logger.info(f"Found {len(buried_spheres)} buried spheres")

    if not buried_spheres:
        logger.info("No buried spheres found. Returning empty pocket list.")
        return []

    # 3. Label spheres as confident or non-confident and filter (if plddt_cutoff provided)
    spheres_to_cluster = buried_spheres
    if plddt_cutoff is not None:
        buried_spheres = label_confidence(
            buried_spheres, atomarray, plddt_cutoff=plddt_cutoff
        )
        spheres_to_cluster = filter_low_confidence_spheres(buried_spheres)
        logger.info(f"Found {len(spheres_to_cluster)} confident spheres")
        if not spheres_to_cluster:
            logger.info("No confident spheres found. Returning empty pocket list.")
            return []

    # 4. Cluster spheres into pockets
    pocket_clusters = cluster_spheres(
        spheres_to_cluster, merge_distance=merge_distance, min_spheres=min_spheres
    )
    logger.info(f"Found {len(pocket_clusters)} clusters (potential pockets)")

    if not pocket_clusters:
        logger.info("No clusters found. Returning empty pocket list.")
        return []

    # Create Pocket objects with descriptors
    pockets = [
        create_pocket(pocket_id, cluster)
        for pocket_id, cluster in enumerate(pocket_clusters)
    ]

    # Sort by score (descending)
    pockets.sort(key=lambda p: p.score, reverse=True)
    logger.info(f"Pocket detection complete. {len(pockets)} pockets found")

    return pockets


def find_confident_pockets(
    atomarray: struc.AtomArray,
    *,
    r_min: float = 3.0,
    r_max: float = 6.0,
    polar_probe_radius: float = 1.8,
    sasa_cutoff: float = 1.0,
    plddt_cutoff: float = 70.0,
    merge_distance: float = 1.75,
    min_spheres: int = 35,
    ignore_hydrogens: bool = True,
    ignore_water: bool = True,
    ignore_hetero: bool = True,
) -> list[Pocket]:
    """Detect binding pockets in a protein structure with pLDDT-based confidence filtering.

    Convenience wrapper around find_pockets() that applies pLDDT-based confidence filtering.

    Args:
        atomarray: Biotite AtomArray with structure data
        r_min: Minimum alpha-sphere radius (Å). Default: 3.0
        r_max: Maximum alpha-sphere radius (Å). Default: 6.0
        polar_probe_radius: Radius to test atom contact for polarity (Å). Default: 1.8
        sasa_cutoff: Solvent accessible surface area cutoff for a buried spheres. Default: 1.0
        plddt_cutoff: AlphaFold pLDDT cutoff for confident spheres. Default: 70.0
        merge_distance: Distance threshold for merging nearby sphere clusters (Å). Default: 1.75
        min_spheres: Minimum number of spheres per pocket cluster. Default: 35
        ignore_hydrogens: Ignore hydrogen atoms (recommended). Default: True
        ignore_water: Ignore water molecules (recommended). Default: True
        ignore_hetero: Ignore hetero atoms i.e. ligands (recommended). Default: True

    Returns:
        pockets: list of detected Pocket objects, sorted by score (highest first)

    Raises:
        ValidationError: if atomarray is invalid or empty
        ValueError: if parameters are invalid

    Examples:
        >>> import pocketeer as pt
        >>> atomarray = pt.load_structure("protein.pdb")
        >>> pockets = pt.find_confident_pockets(atomarray)

        >>> # With custom parameters
        >>> pockets = pt.find_confident_pockets(atomarray, r_min=2.5, r_max=7.0, min_spheres=25)
    """
    return find_pockets(
        atomarray,
        r_min=r_min,
        r_max=r_max,
        polar_probe_radius=polar_probe_radius,
        sasa_cutoff=sasa_cutoff,
        plddt_cutoff=plddt_cutoff,
        merge_distance=merge_distance,
        min_spheres=min_spheres,
        ignore_hydrogens=ignore_hydrogens,
        ignore_water=ignore_water,
        ignore_hetero=ignore_hetero,
    )


def find_globular_pockets(
    atomarray: struc.AtomArray,
    *,
    r_min: float = 3.0,
    r_max: float = 6.0,
    polar_probe_radius: float = 5.0,
    sasa_cutoff: float = 1.0,
    plddt_cutoff: float | None = None,
    merge_distance: float = 1.75,
    min_spheres: int = 35,
    centroid_radius: float = 15.0,
    min_atoms_in_radius: int = 100,
    ignore_hydrogens: bool = True,
    ignore_water: bool = True,
    ignore_hetero: bool = True,
) -> list[Pocket]:
    """Detect globular binding pockets with sufficient local atom density.

    Filters pockets based on the number of atoms within a specified radius
    of the pocket centroid. This helps identify pockets in globular regions
    with sufficient structural context.

    Args:
        atomarray: Biotite AtomArray with structure data
        r_min: Minimum alpha-sphere radius (Å). Default: 3.0
        r_max: Maximum alpha-sphere radius (Å). Default: 6.0
        polar_probe_radius: Radius to test atom contact for polarity (Å). Default: 5.0
        sasa_cutoff: Solvent accessible surface area cutoff for buried spheres (Å²). Default: 1.0
        plddt_cutoff: AlphaFold pLDDT cutoff for confident spheres. Default: None (no filtering)
        merge_distance: Distance threshold for merging nearby sphere clusters (Å). Default: 1.75
        min_spheres: Minimum number of spheres per pocket cluster. Default: 35
        centroid_radius: Radius around pocket centroid to count atoms (Å). Default: 15.0
        min_atoms_in_radius: Minimum number of atoms within centroid_radius to keep pocket. Default: 100
        ignore_hydrogens: Ignore hydrogen atoms (recommended). Default: True
        ignore_water: Ignore water molecules (recommended). Default: True
        ignore_hetero: Ignore hetero atoms i.e. ligands (recommended). Default: True

    Returns:
        list[Pocket]: List of detected globular Pocket objects that meet the atom density
            criteria, sorted by score (highest first)

    Examples:
        >>> import pocketeer as pt
        >>> atomarray = pt.load_structure("protein.pdb")
        >>> pockets = pt.find_globular_pockets(atomarray)

        >>> # With custom density parameters
        >>> pockets = pt.find_globular_pockets(
        ...     atomarray,
        ...     centroid_radius=20.0,
        ...     min_atoms_in_radius=150
        ... )
    """
    # Find all pockets using standard detection
    pockets = find_pockets(
        atomarray,
        r_min=r_min,
        r_max=r_max,
        polar_probe_radius=polar_probe_radius,
        sasa_cutoff=sasa_cutoff,
        plddt_cutoff=plddt_cutoff,
        merge_distance=merge_distance,
        min_spheres=min_spheres,
        ignore_hydrogens=ignore_hydrogens,
        ignore_water=ignore_water,
        ignore_hetero=ignore_hetero,
    )

    if not pockets:
        logger.info("No pockets found. Returning empty list.")
        return []

    # Build KDTree for efficient spatial queries
    tree = KDTree(atomarray.coord)

    # Filter pockets based on local atom density
    globular_pockets = []
    for pocket in pockets:
        # Count atoms within specified radius of pocket centroid
        atom_indices_in_radius = tree.query_ball_point(
            pocket.centroid, r=centroid_radius
        )
        atom_count = len(atom_indices_in_radius)

        # Keep pocket if it meets the minimum atom density requirement
        if atom_count >= min_atoms_in_radius:
            globular_pockets.append(pocket)

    logger.info(
        f"Found {len(globular_pockets)} globular pockets out of {len(pockets)} total pockets"
    )

    return globular_pockets
