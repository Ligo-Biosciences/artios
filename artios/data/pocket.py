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
        sphere.is_buried = np.all(defining_atom_sasa <= sasa_cutoff)

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
        sphere.is_confident = np.all(defining_atom_plddt >= plddt_cutoff)

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


# Amino acid categories for binding site analysis
_AROMATIC_RES = frozenset({"PHE", "TYR", "TRP", "HIS"})
_CHARGED_RES = frozenset({"ASP", "GLU", "LYS", "ARG", "HIS"})

# Binding score model weights (logistic regression trained on 130 PDB structures
# with ground-truth ligand positions from pdb-samples/tracker.json).
# Trained with 0.2Å Gaussian coordinate noise augmentation for robustness.
# Features are: n_spheres, lining_residues, enclosure, mean_radius,
# aromatic_ratio, charged_ratio, residue_diversity, carbon_atom_ratio
_BINDING_WEIGHTS = np.array(
    [0.004118, 0.021214, 6.494799, 2.550456, 3.697266, -1.462200, 0.122680, 4.379412]
)
_BINDING_BIAS = -21.796847


def _compute_binding_features(
    pocket: Pocket,
    protein_atoms: struc.AtomArray,
    tree: KDTree,
) -> np.ndarray:
    """Compute binding-relevant features for a pocket.

    Returns feature vector: [n_spheres, lining_residues, enclosure,
    mean_radius, aromatic_ratio, charged_ratio, residue_diversity,
    carbon_atom_ratio]
    """
    sphere_centers = np.array([s.center for s in pocket.spheres])
    centroid = pocket.centroid

    # --- Contact analysis (4.5Å from sphere centers) ---
    contact_indices: set[int] = set()
    for sc in sphere_centers:
        contact_indices.update(tree.query_ball_point(sc, 4.5))

    if contact_indices:
        contact_atoms = protein_atoms[sorted(contact_indices)]
        res_keys = set(
            zip(contact_atoms.chain_id, contact_atoms.res_id, contact_atoms.res_name)
        )
        lining_residues = len(res_keys)
        residue_diversity = len({r[2] for r in res_keys})
        aromatic_ratio = (
            sum(1 for r in res_keys if r[2] in _AROMATIC_RES) / lining_residues
        )
        charged_ratio = (
            sum(1 for r in res_keys if r[2] in _CHARGED_RES) / lining_residues
        )
        carbon_atom_ratio = float(
            np.sum(contact_atoms.element == "C") / len(contact_atoms)
        )
    else:
        lining_residues = 0
        residue_diversity = 0
        aromatic_ratio = 0.0
        charged_ratio = 0.0
        carbon_atom_ratio = 0.0

    # --- Enclosure (ray casting from centroid) ---
    n_rays = 100
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_rays)
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_rays)
    directions = np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )
    steps = np.arange(0.5, 10.5, 0.5)
    hits = 0
    for d in directions:
        ray_points = centroid + np.outer(steps, d)
        dists, _ = tree.query(ray_points)
        if np.any(dists < 2.0):
            hits += 1
    enclosure = hits / n_rays

    # Mean sphere radius
    mean_radius = float(np.mean([s.radius for s in pocket.spheres]))

    return np.array(
        [
            pocket.n_spheres,
            lining_residues,
            enclosure,
            mean_radius,
            aromatic_ratio,
            charged_ratio,
            residue_diversity,
            carbon_atom_ratio,
        ]
    )


def score_binding_pocket(
    pocket: Pocket,
    protein_atoms: struc.AtomArray,
    tree: KDTree,
) -> float:
    """Score how likely a pocket is a realistic ligand binding site.

    Uses a logistic model trained on 130 PDB structures with known ligand
    positions and 0.2A Gaussian coordinate noise augmentation. Higher scores
    indicate more realistic pockets with characteristics of real binding sites:
    appropriate size, good enclosure, aromatic contacts, and diverse residue
    environments.

    Args:
        pocket: Detected pocket to score
        protein_atoms: Protein-only atoms (no water, hetero, or hydrogens)
        tree: KDTree built from protein_atoms.coord

    Returns:
        Score between 0 and 1 (probability of being a real binding site)
    """
    features = _compute_binding_features(pocket, protein_atoms, tree)
    logit = float(np.dot(_BINDING_WEIGHTS, features) + _BINDING_BIAS)
    return 1.0 / (1.0 + np.exp(-logit))


def find_realistic_pockets(
    atomarray: struc.AtomArray,
    *,
    r_min: float = 3.0,
    r_max: float = 6.0,
    polar_probe_radius: float = 5.0,
    sasa_cutoff: float = 10.0,
    plddt_cutoff: float | None = None,
    merge_distance: float = 1.75,
    min_spheres: int = 35,
    min_binding_score: float = 0.15,
    ignore_hydrogens: bool = True,
    ignore_water: bool = True,
    ignore_hetero: bool = True,
) -> list[Pocket]:
    """Detect realistic ligand binding pockets in a protein structure.

    Extends find_pockets() with a binding-likelihood score that filters out
    surface clefts and non-functional cavities. Each pocket is scored by a
    logistic model using 8 features: pocket size (n_spheres), contact complexity
    (lining_residues, residue_diversity), geometry (enclosure, mean_radius),
    and chemistry (aromatic_ratio, charged_ratio, carbon_atom_ratio).
    Trained with 0.2A Gaussian coordinate noise augmentation for robustness.

    Validated on a held-out test set of 39 PDBs:
        - At min_binding_score=0.15 (default): ~83% recall, ~36% precision
        - At min_binding_score=0.25: ~64% recall, ~46% precision

    Args:
        atomarray: Biotite AtomArray with structure data
        r_min: Minimum alpha-sphere radius (Å). Default: 3.0
        r_max: Maximum alpha-sphere radius (Å). Default: 6.0
        polar_probe_radius: Probe radius for SASA polarity test (Å). Default: 5.0
        sasa_cutoff: SASA cutoff for buried spheres (Å²). Default: 10.0
        plddt_cutoff: AlphaFold pLDDT cutoff. None disables filtering. Default: None
        merge_distance: Distance for merging sphere clusters (Å). Default: 1.75
        min_spheres: Minimum spheres per pocket cluster. Default: 35
        min_binding_score: Minimum binding score to keep (0-1). Default: 0.15
        ignore_hydrogens: Ignore hydrogen atoms. Default: True
        ignore_water: Ignore water molecules. Default: True
        ignore_hetero: Ignore hetero atoms (ligands). Default: True

    Returns:
        list[Pocket]: Pockets likely to be real binding sites, sorted by
            binding_score (highest first). Each pocket has a `binding_score`
            attribute set.
    """
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
        return []

    # Build protein-only atom array and KDTree for scoring
    protein_atoms = atomarray[~atomarray.hetero]
    protein_atoms = protein_atoms[protein_atoms.element != "H"]
    protein_atoms = protein_atoms[protein_atoms.res_name != "HOH"]
    tree = KDTree(protein_atoms.coord)

    # Score and filter
    binding_pockets = []
    for pocket in pockets:
        pocket.binding_score = score_binding_pocket(pocket, protein_atoms, tree)
        if pocket.binding_score >= min_binding_score:
            binding_pockets.append(pocket)

    binding_pockets.sort(key=lambda p: p.binding_score, reverse=True)
    logger.info(
        f"Found {len(binding_pockets)} binding pockets out of {len(pockets)} "
        f"total (min_binding_score={min_binding_score})"
    )
    return binding_pockets


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
