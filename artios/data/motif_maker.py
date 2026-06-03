"""Stochastic motif sampler for model training.

Attached to an ``AnnotatedStructure`` via ``structure.with_sampled_motif(maker)``;
returns a ``MotifAnnotation`` describing which atoms belong to the motif.

The branches of ``sample()``:

- ``_generate_ligand_motif_mask`` — centred on a ligand. Used on ligand complexes
  with probability ``ligand_motif_prob``.
- ``_generate_pocket_motif_mask`` — centred on a detected pocket. Used with
  probability ``pocket_motif_prob`` when pocket annotations are present.
- ``_generate_chain_motif_mask`` — fallback, centred on a random polymer atom.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from biotite.structure import AtomArray

from artios.common import atom_types
from artios.data.structure.annotation import MotifAnnotation

if TYPE_CHECKING:
    from artios.data.structure.structure import AnnotatedStructure

logger = getLogger(__name__)


ProbabilityTable = tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class MotifMakerConfig:
    """Fixed motif-sampling recipe for motif-scaffolding training."""

    # --- Branch selection ----------------------------------------------------
    ligand_motif_prob: float = 0.9
    pocket_motif_prob: float = 1.0

    # --- Ligand motif --------------------------------------------------------
    whole_ligand_prob: float = 0.0
    drop_ligand_motif_prob: float = 0.1
    complex_ligand_threshold: int = 5
    complex_ligand_preference_prob: float = 1.0
    ligand_region_count_probs: ProbabilityTable = field(
        default_factory=lambda: ((1, 0.5), (2, 0.4), (3, 0.07), (4, 0.03))
    )
    ligand_bfs_atom_count_probs: ProbabilityTable = field(
        default_factory=lambda: (
            (1, 0.15),
            (2, 0.2),
            (3, 0.25),
            (4, 0.2),
            (5, 0.125),
            (6, 0.075),
        )
    )
    ligand_fixed_size_bfs_prob: float = 0.9
    supplementary_ligand_distance: float = 5.0
    supplementary_ligand_prob: float = 0.5

    # --- Around-a-point mask (shared by all branches) ------------------------
    r_min: float = 5.0
    r_max: float = 8.0
    residue_inclusion_prob: float = 0.8
    use_tip_atoms_only_prob: float = 0.95
    whole_sidechain_prob: float = 0.0
    max_motif_residues: int = 15

    # --- Partial residue (sidechain BFS) -------------------------------------
    sidechain_seed_is_tip_prob: float = 0.0
    sidechain_seed_is_in_sphere_prob: float = 1.0
    sidechain_bfs_depth_prob: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ligand_region_count_probs",
            _normalize_probability_table(self.ligand_region_count_probs),
        )
        object.__setattr__(
            self,
            "ligand_bfs_atom_count_probs",
            _normalize_probability_table(self.ligand_bfs_atom_count_probs),
        )
        _validate_probability_config(self)


@dataclass(slots=True, kw_only=True)
class MotifMaker:
    """Stochastic motif sampler. See module docstring for branch descriptions."""

    config: MotifMakerConfig = field(default_factory=MotifMakerConfig)
    _region_counts: np.ndarray = field(init=False, repr=False)
    _region_probs: np.ndarray = field(init=False, repr=False)
    _bfs_counts: np.ndarray = field(init=False, repr=False)
    _bfs_probs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._region_counts, self._region_probs = _dist_to_arrays(
            self.config.ligand_region_count_probs
        )
        self._bfs_counts, self._bfs_probs = _dist_to_arrays(
            self.config.ligand_bfs_atom_count_probs
        )

    def sample(self, structure: AnnotatedStructure) -> MotifAnnotation:
        """Sample a motif overlay from a loaded structure."""
        aa = structure.to_atom_array()

        # TODO: delete this check once we have pocket annotations for the PDB
        has_pocket_id = "pocket_id" in aa.get_annotation_categories()

        if np.random.rand() < self.config.ligand_motif_prob and aa.is_ligand.any():
            is_motif = self._generate_ligand_motif_mask(aa)
        elif (
            has_pocket_id
            and np.random.rand() < self.config.pocket_motif_prob
            and aa.pocket_id.any()
        ):
            is_motif = self._generate_pocket_motif_mask(aa)
        else:
            is_motif = self._generate_chain_motif_mask(aa)

        if is_motif.sum() == 1:
            is_motif = np.zeros(len(aa), dtype=int)
        elif is_motif.any():
            num_residues = len(np.unique(aa[is_motif > 0].res_id))
            if num_residues > self.config.max_motif_residues:
                is_motif = np.zeros(len(aa), dtype=int)

        is_motif_related = self._get_is_motif_related_mask(aa, is_motif)
        return MotifAnnotation(
            is_motif=is_motif.astype(int),
            is_motif_related=is_motif_related.astype(int),
        )

    def _get_is_motif_related_mask(
        self, atom_array: AtomArray, is_motif: np.ndarray
    ) -> np.ndarray:
        """Mask of atoms sharing a residue (by ``ref_space_uid``) with a motif atom.

        Includes the motif atoms themselves. Takes ``is_motif`` explicitly so
        the caller need not have stamped it onto ``atom_array`` first.
        """
        is_motif_related = np.zeros(len(atom_array), dtype=int)
        unique_ref_space_uids = np.unique(atom_array.ref_space_uid[is_motif > 0])
        is_motif_related[np.isin(atom_array.ref_space_uid, unique_ref_space_uids)] = 1
        return is_motif_related

    def _sample_bfs_atom_count(self, ligand_graph_size: int) -> int:
        """Number of atoms to collect via BFS for a ligand region.

        With probability ``ligand_fixed_size_bfs_prob``, samples from
        ``ligand_bfs_atom_count_probs``; otherwise proportion-based.
        """
        if np.random.rand() < self.config.ligand_fixed_size_bfs_prob:
            return int(np.random.choice(self._bfs_counts, p=self._bfs_probs))
        ligand_proportion = np.random.rand()
        return max(1, int(ligand_proportion * ligand_graph_size))

    def _add_supplementary_ligands(
        self,
        atom_array: AtomArray,
        seed_coord: np.ndarray,
        exclude_asym_id: str,
    ) -> np.ndarray:
        """
        Find and add supplementary ligands near the seed point to the motif.

        For each ligand chain (other than exclude_asym_id) with atoms within
        supplementary_ligand_distance of seed_coord, with probability supplementary_ligand_prob,
        performs a BFS from the closest atom and adds those atoms to the motif.

        Args:
            atom_array (AtomArray): Full atom array.
            seed_coord (np.ndarray): Coordinates of the seed point.
            exclude_asym_id (str): The main ligand's asym_id to exclude from consideration.

        Returns:
            np.ndarray: Updated motif mask with supplementary ligand atoms added.
        """
        supplementary_ligand_mask = np.zeros(len(atom_array), dtype=int)

        # Find all ligand atoms (excluding the main ligand)
        ligand_mask = atom_array.chain_mol_type == "ligand"
        other_ligand_mask = ligand_mask & (atom_array.asym_id_int != exclude_asym_id)

        if not other_ligand_mask.any():
            return supplementary_ligand_mask

        # Calculate distances from seed point to all other ligand atoms
        other_ligand_indices = np.where(other_ligand_mask)[0]
        other_ligand_coords = atom_array.coord[other_ligand_indices]
        distances = np.linalg.norm(other_ligand_coords - seed_coord, axis=-1)

        # Find atoms within distance threshold
        within_distance_mask = distances <= self.config.supplementary_ligand_distance
        if not within_distance_mask.any():
            return supplementary_ligand_mask

        # Get unique ligand chains that have atoms within distance
        nearby_indices = other_ligand_indices[within_distance_mask]
        nearby_distances = distances[within_distance_mask]
        nearby_asym_ids = atom_array.asym_id_int[nearby_indices]
        unique_nearby_asym_ids = np.unique(nearby_asym_ids)

        # Process each nearby ligand chain independently
        for asym_id in unique_nearby_asym_ids:
            if np.random.rand() >= self.config.supplementary_ligand_prob:
                continue

            # Find the closest atom from this chain
            chain_mask_in_nearby = nearby_asym_ids == asym_id
            chain_distances = nearby_distances[chain_mask_in_nearby]
            chain_indices = nearby_indices[chain_mask_in_nearby]
            closest_idx = chain_indices[np.argmin(chain_distances)]

            # Get the full ligand chain for BFS
            full_chain_mask = (atom_array.asym_id_int == asym_id) & ligand_mask
            chain_array = atom_array[full_chain_mask]
            chain_graph = chain_array.bonds.as_graph()
            graph_nodes = list(chain_graph.nodes())

            if full_chain_mask.sum() == 1 or len(graph_nodes) == 0:
                supplementary_ligand_mask[closest_idx] = 1
                continue

            # Find the local index of the closest atom in the chain
            full_chain_indices = np.where(full_chain_mask)[0]
            local_closest_idx = np.where(full_chain_indices == closest_idx)[0]
            if len(local_closest_idx) == 0 or local_closest_idx[0] not in graph_nodes:
                continue
            else:
                seed_node = local_closest_idx[0]

            # Sample BFS size and perform BFS
            num_atoms = self._sample_bfs_atom_count(len(chain_graph))
            nodes = bfs_nodes_by_count(chain_graph, seed_node, num_atoms)

            # Add to motif
            selected_indices = full_chain_indices[nodes]
            supplementary_ligand_mask[selected_indices] = 1

        return supplementary_ligand_mask

    def _generate_ligand_motif_mask(self, atom_array: AtomArray) -> np.ndarray:
        """
        Generates a motif mask centered around a ligand.

        Uses two-stage selection to bias towards complex ligands:
        1. Classify ligands as "complex" (>= threshold atoms) or "simple"
        2. With probability complex_ligand_preference_prob, sample from complex pool
        3. Otherwise sample uniformly from all ligands

        For partial ligand motifs, supports multiple non-contiguous regions:
        - Number of regions sampled from ligand_region_count_probs (1-4 regions)
        - Each region's size sampled from ligand_bfs_atom_count_probs (1-5 atoms)
        - Seed atoms for regions are non-adjacent (graph distance >= 2)

        Args:
            atom_array (AtomArray): Input Biotite AtomArray

        Returns:
            np.ndarray: Integer mask (0/1) indicating motif atoms
        """
        is_motif = np.zeros(len(atom_array), dtype=int)

        # Select a ligand using two-stage selection
        ligand_mask = atom_array.chain_mol_type == "ligand"
        ligand_asym_ids = np.unique(atom_array.asym_id_int[ligand_mask])

        # Classify ligands by complexity (atom count)
        complex_ligand_ids = []
        for asym_id in ligand_asym_ids:
            atom_count = (atom_array.asym_id_int == asym_id).sum()
            if atom_count >= self.config.complex_ligand_threshold:
                complex_ligand_ids.append(asym_id)

        # Prefer complex ligands if available
        if (
            len(complex_ligand_ids) > 0
            and np.random.rand() < self.config.complex_ligand_preference_prob
        ):
            ligand_motif_asym_id = np.random.choice(complex_ligand_ids)
        else:
            ligand_motif_asym_id = np.random.choice(ligand_asym_ids)

        ligand_motif_chain_mask = (
            atom_array.asym_id_int == ligand_motif_asym_id
        ) & ligand_mask

        if (
            np.random.rand() < self.config.whole_ligand_prob
            or ligand_motif_chain_mask.sum() == 1
        ):
            ligand_atoms = atom_array[ligand_motif_chain_mask]
            seed_coord = get_random_seed_atom_from_array(ligand_atoms)

            surrounding_mask = self._generate_motif_around_point_mask(
                point_coord=seed_coord,
                atom_array=atom_array,
            )

        else:
            # Add partial ligand to motif with potentially multiple non-contiguous regions
            ligand_array = atom_array[ligand_motif_chain_mask]
            ligand_graph = ligand_array.bonds.as_graph()
            graph_nodes = list(ligand_graph.nodes())

            if len(graph_nodes) == 0:
                return is_motif

            num_regions = int(
                np.random.choice(self._region_counts, p=self._region_probs)
            )

            seed_atoms = select_non_adjacent_seeds(
                ligand_graph, graph_nodes, num_regions
            )

            # Collect all motif nodes from all regions
            all_motif_nodes: list[int] = []
            for seed_atom in seed_atoms:
                num_atoms = self._sample_bfs_atom_count(len(ligand_graph))
                nodes = bfs_nodes_by_count(ligand_graph, seed_atom, num_atoms)
                all_motif_nodes.extend(nodes)

            # Remove duplicates
            all_motif_nodes = list(set(all_motif_nodes))

            # Mark these ligand atoms as motif
            full_ligand_indices = np.where(ligand_motif_chain_mask)[0]
            ligand_motif_chain_mask = full_ligand_indices[all_motif_nodes]

            # Get seed point from partial ligand
            partial_ligand_atoms = atom_array[ligand_motif_chain_mask]
            seed_coord = get_random_seed_atom_from_array(partial_ligand_atoms)

            # Add surrounding residues
            surrounding_mask = self._generate_motif_around_point_mask(
                point_coord=seed_coord,
                atom_array=atom_array,
            )

        if surrounding_mask.sum() == 0:
            # Avoid returning motif containing solely ligand.
            return np.zeros(len(atom_array), dtype=int)

        is_motif[surrounding_mask > 0] = 1

        if np.random.rand() < self.config.drop_ligand_motif_prob:
            is_motif[ligand_motif_chain_mask] = 0
        else:
            is_motif[ligand_motif_chain_mask] = 1

        # Add supplementary ligands near seed point
        supplementary_ligand_mask = self._add_supplementary_ligands(
            atom_array, seed_coord, ligand_motif_asym_id
        )

        is_motif[supplementary_ligand_mask > 0] = 1

        return is_motif

    def _generate_pocket_motif_mask(self, atom_array: AtomArray) -> np.ndarray:
        """
        Generates a motif mask centered around a pocket.

        Args:
            atom_array (AtomArray): Input Biotite AtomArray

        Returns:
            np.ndarray: Integer mask (0/1) indicating motif atoms
        """
        is_motif = np.zeros(len(atom_array), dtype=int)

        pocket_ids = np.unique(atom_array.pocket_id)
        pocket_ids = pocket_ids[pocket_ids > 0]
        random_pocket_id = np.random.choice(pocket_ids)

        pocket_mask = atom_array.pocket_id == random_pocket_id

        # Get the centroid of the pocket
        pocket_atoms = atom_array[pocket_mask]
        pocket_centroid = np.mean(pocket_atoms.coord, axis=0)

        # Generate motif around the centroid
        motif_mask = self._generate_motif_around_point_mask(
            point_coord=pocket_centroid,
            atom_array=atom_array,
            optional_mask=pocket_mask,
        )
        is_motif[motif_mask > 0] = 1

        return is_motif

    def _generate_chain_motif_mask(self, atom_array: AtomArray) -> np.ndarray:
        """
        Generates a motif mask centered around a random atom on a polymer chain.

        Args:
            atom_array (AtomArray): Input Biotite AtomArray

        Returns:
            np.ndarray: Integer mask (0/1) indicating motif atoms
        """
        is_motif = np.zeros(len(atom_array), dtype=int)

        # Filter for polymer chains
        polymer_mask = np.isin(atom_array.chain_mol_type, ["protein", "RNA", "DNA"])

        if not polymer_mask.any():
            return is_motif

        # Pick a random polymer chain
        polymer_atoms = atom_array[polymer_mask]
        chain_id = np.random.choice(np.unique(polymer_atoms.asym_id_int))

        chain_mask = atom_array.asym_id_int == chain_id
        chain_array = atom_array[chain_mask]

        # Get seed point from a random atom on the chain
        seed_coord = get_random_seed_atom_from_array(chain_array)

        # Generate motif around seed
        motif_mask = self._generate_motif_around_point_mask(
            point_coord=seed_coord,
            atom_array=chain_array,
        )

        # Map back to full array indices
        chain_indices = np.where(chain_mask)[0]
        full_motif_mask = np.zeros(len(atom_array), dtype=int)
        full_motif_mask[chain_indices[motif_mask > 0]] = 1

        return full_motif_mask

    def _generate_motif_around_point_mask(
        self,
        point_coord: np.ndarray,
        atom_array: AtomArray,
        optional_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generates a motif mask around a point by identifying residues within a sphere.

        Args:
            point_coord (np.ndarray): Center point coordinates
            atom_array (AtomArray): AtomArray to generate motif from
            optional_mask (np.ndarray | None, optional): Optional boolean mask to further filter atoms.
                If provided, will be combined with atoms_in_sphere_mask using logical AND. Defaults to None.

        Returns:
            np.ndarray: Integer mask (0/1) indicating motif atoms
        """
        if np.random.rand() < self.config.use_tip_atoms_only_prob:
            use_tip_atoms_only = True
        else:
            use_tip_atoms_only = False

        is_motif = np.zeros(len(atom_array), dtype=int)

        # Find polymer atoms within sphere
        r = np.random.uniform(self.config.r_min, self.config.r_max)

        distances = np.linalg.norm(atom_array.coord - point_coord, axis=-1)
        within_radius_mask = distances <= r

        polymer_mask = np.isin(atom_array.chain_mol_type, ["protein", "RNA", "DNA"])
        atoms_in_sphere_mask = within_radius_mask & polymer_mask

        # Apply optional mask
        if optional_mask is not None:
            atoms_in_sphere_mask = atoms_in_sphere_mask & optional_mask

        if not atoms_in_sphere_mask.any():
            return is_motif

        # Get unique residues in sphere
        if use_tip_atoms_only:
            # Create mask for atoms that are valid motif atoms based on residue type
            is_valid_tip_atom = np.zeros(len(atom_array), dtype=bool)
            for res_name, valid_atoms in atom_types.TIP_ATOMS.items():
                res_mask = atom_array.res_name == res_name
                atom_mask = np.isin(atom_array.atom_name, valid_atoms)
                is_valid_tip_atom |= res_mask & atom_mask
            atoms_in_sphere_mask = atoms_in_sphere_mask & is_valid_tip_atom

        atoms_in_sphere = atom_array[atoms_in_sphere_mask]
        unique_ref_space_uids = np.unique(atoms_in_sphere.ref_space_uid)

        if len(unique_ref_space_uids) == 0:
            return is_motif

        # Process each residue
        for uid in unique_ref_space_uids:
            # Skip residue with probability
            if np.random.rand() > self.config.residue_inclusion_prob:
                continue

            # Get all atoms of this residue
            residue_mask = atom_array.ref_space_uid == uid

            if np.random.rand() > self.config.whole_sidechain_prob:
                # Partial residue
                partial_mask = self._get_partial_residue_motif_mask(
                    atom_array, residue_mask, in_sphere_mask=atoms_in_sphere_mask
                )
                is_motif[partial_mask > 0] = 1
            else:
                # Whole residue
                is_motif[residue_mask > 0] = 1

        # Filter the final motif for only
        if use_tip_atoms_only:
            is_motif = is_motif * is_valid_tip_atom.astype(int)

        return is_motif

    def _get_partial_residue_motif_mask(
        self,
        atom_array: AtomArray,
        residue_mask: np.ndarray,
        in_sphere_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Gets partial residue motif mask using BFS on the residue bond graph.

        Args:
            atom_array (AtomArray): Full atom array
            residue_mask (np.ndarray): Boolean mask indicating which atoms belong to the residue
            in_sphere_mask (np.ndarray | None, optional): Boolean mask indicating which atoms are in the sphere.
        Returns:
            np.ndarray: Integer mask (0/1) indicating which atoms of the residue are in the motif
        """
        partial_motif_mask = np.zeros(len(atom_array), dtype=int)

        residue_array = atom_array[residue_mask]
        if len(residue_array) == 0:
            return partial_motif_mask

        G_res = residue_array.bonds.as_graph()
        graph_nodes = list(G_res.nodes())

        if len(graph_nodes) == 0:
            return partial_motif_mask

        # Pick seed atom
        p = np.random.rand()
        residue_indices = np.where(residue_mask)[0]
        local_in_sphere_mask = in_sphere_mask[residue_indices]
        local_indices_in_sphere = np.where(local_in_sphere_mask)[0]

        # Filter to valid graph nodes
        graph_node_set = set(graph_nodes)
        valid_seed_atoms = [
            idx for idx in local_indices_in_sphere if idx in graph_node_set
        ]

        if p < self.config.sidechain_seed_is_in_sphere_prob and valid_seed_atoms:
            seed_atom = np.random.choice(valid_seed_atoms)
        elif (
            p
            < self.config.sidechain_seed_is_in_sphere_prob
            + self.config.sidechain_seed_is_tip_prob
        ):
            seed_atom = graph_nodes[
                -1
            ]  # TODO: I don't think the last node in the graph is neccessarily the tip
        else:
            seed_atom = np.random.choice(graph_nodes)

        # Pick depth
        num_bonds = np.random.geometric(self.config.sidechain_bfs_depth_prob) - 1

        # Get atoms within num_bonds of seed
        motif_atom_ids = nx.single_source_shortest_path_length(
            G_res, seed_atom, cutoff=num_bonds
        )
        motif_atom_ids = list(motif_atom_ids.keys())

        # Map back to full array
        residue_indices = np.where(residue_mask)[0]
        selected_indices = residue_indices[motif_atom_ids]
        partial_motif_mask[selected_indices] = 1

        return partial_motif_mask


def _normalize_probability_table(
    table: Mapping[int | str, float] | Iterable[tuple[int | str, float]],
) -> ProbabilityTable:
    items = table.items() if isinstance(table, Mapping) else table
    return tuple((int(value), float(prob)) for value, prob in items)


def _validate_probability_config(config: MotifMakerConfig) -> None:
    scalar_prob_names = (
        "ligand_motif_prob",
        "pocket_motif_prob",
        "whole_ligand_prob",
        "drop_ligand_motif_prob",
        "complex_ligand_preference_prob",
        "ligand_fixed_size_bfs_prob",
        "supplementary_ligand_prob",
        "residue_inclusion_prob",
        "use_tip_atoms_only_prob",
        "whole_sidechain_prob",
        "sidechain_seed_is_tip_prob",
        "sidechain_seed_is_in_sphere_prob",
        "sidechain_bfs_depth_prob",
    )
    for name in scalar_prob_names:
        value = getattr(config, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    if config.r_min > config.r_max:
        raise ValueError("r_min must be less than or equal to r_max")
    if config.complex_ligand_threshold <= 0:
        raise ValueError("complex_ligand_threshold must be greater than 0")
    if config.max_motif_residues <= 0:
        raise ValueError("max_motif_residues must be greater than 0")

    _validate_probability_table(
        "ligand_region_count_probs", config.ligand_region_count_probs
    )
    _validate_probability_table(
        "ligand_bfs_atom_count_probs", config.ligand_bfs_atom_count_probs
    )


def _validate_probability_table(name: str, table: ProbabilityTable) -> None:
    if not table:
        raise ValueError(f"{name} must not be empty")
    total = sum(prob for _, prob in table)
    if not np.isclose(total, 1.0):
        raise ValueError(f"{name} probabilities must sum to 1")
    for value, prob in table:
        if value <= 0:
            raise ValueError(f"{name} values must be positive")
        if prob < 0:
            raise ValueError(f"{name} probabilities must be non-negative")


def _dist_to_arrays(dist: ProbabilityTable) -> tuple[np.ndarray, np.ndarray]:
    """Convert a probability table to parallel arrays for ``np.random.choice``."""
    values, probs = zip(*dist)
    return np.array(values), np.array(probs)


DEFAULT_MOTIF_MAKER_CONFIG = MotifMakerConfig()


def get_random_seed_atom_from_array(atom_array: AtomArray) -> np.ndarray:
    """
    Picks a seed atom uniformly at random from the input array
    and returns its coordinates.

    Args:
        atom_array (AtomArray): Input atom array

    Returns:
        np.ndarray: Coordinates of the randomly selected seed atom [..., 3]
    """
    random_idx = np.random.randint(len(atom_array))
    return atom_array[random_idx].coord


def bfs_nodes_by_count(G: nx.Graph, source: Any, target_count: int) -> list[Any]:
    """
    Perform BFS from source and return nodes up to target_count.

    Args:
        G (nx.Graph): The graph to traverse.
        source (Any): Starting node for BFS.
        target_count (int): Number of nodes to collect.

    Returns:
        list: List of nodes in BFS order, up to target_count.
    """
    assert target_count > 0, "target_count must be greater than 0"
    visited = [source]
    for u, v in nx.bfs_edges(G, source):
        if len(visited) >= target_count:
            break
        visited.append(v)

    return visited


def select_non_adjacent_seeds(
    graph: nx.Graph,
    graph_nodes: list[Any],
    num_seeds: int,
) -> list[Any]:
    """
    Select seed atoms from a graph ensuring they are non-adjacent (graph distance >= 2).

    Args:
        graph (nx.Graph): The molecular graph.
        graph_nodes (list[Any]): List of all nodes in the graph.
        num_seeds (int): Desired number of seed atoms to select.

    Returns:
        list[Any]: List of selected seed atom indices. May be fewer than num_seeds
            if not enough non-adjacent nodes are available.
    """
    selected_seeds = []
    available_nodes = set(graph_nodes)

    for _ in range(num_seeds):
        if len(available_nodes) == 0:
            break
        # Pick a random seed from available nodes
        seed = np.random.choice(list(available_nodes))
        selected_seeds.append(seed)

        # Remove the seed and all its neighbors from available nodes
        # This ensures the next seed has graph distance >= 2 from this one
        nodes_to_remove = {seed} | set(graph.neighbors(seed))
        available_nodes -= nodes_to_remove

    return selected_seeds
