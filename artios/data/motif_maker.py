"""
Houses all the logic for generating motifs during model training.

TODO: make the motif maker config its own dataclass / Pydantic object for cleaner access / initialization.
"""

from logging import getLogger
from typing import Any

import networkx as nx
import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import KDTree

from artios.common import atom_types

logger = getLogger(__name__)


class MotifMaker:
    """
    Main class for generating motifs during model training. All methods work with
    integer masks (0/1) to mark which atoms are part of the motif, consistent with
    other masks like is_protein.

    The different motif selection methods are as follows:

    - _generate_ligand_motif_mask:
        Generates a motif mask centered around a ligand. If the complex contains a ligand,
        this method is picked with probability `ligand_motif_prob`. Otherwise,
        _generate_chain_motif_mask is used.

    - _generate_chain_motif_mask:
        Generates a motif mask centered around a pocket-esque point on a polymer chain.

    - _generate_motif_around_point_mask:
        Generates a motif mask centered around a point, identifying residues within a sphere.

    - _get_partial_residue_motif_mask:
        Generates a partial residue motif mask using BFS on the residue bond graph.
    """

    def __init__(
        self,
        ligand_motif_prob: float = 0.8,
        pocket_motif_prob: float = 0.8,
        whole_ligand_prob: float = 0.7,
        drop_ligand_motif_prob: float = 0.1,
        whole_sidechain_prob: float = 0.2,
        pocket_score_threshold: float = 0.18,
        pocket_n_neighbours: int = 25,
        sidechain_seed_is_tip_prob: float = 0.1,
        sidechain_seed_is_in_sphere_prob: float = 0.8,
        sidechain_bfs_depth_prob: float = 0.5,
        residue_inclusion_prob: float = 0.9,
        use_tip_atoms_only_prob: float = 0.9,
        r_min: float = 3.0,
        r_max: float = 6.0,
        max_motif_residues: int = 20,
        motif_train_rate: float = 0.0,
        complex_ligand_threshold: int = 5,
        complex_ligand_preference_prob: float = 0.9,
        ligand_region_count_probs: dict[str, float] = {
            "1": 0.5,
            "2": 0.4,
            "3": 0.07,
            "4": 0.03,
        },
        ligand_bfs_atom_count_probs: dict[str, float] = {
            "1": 0.15,
            "2": 0.25,
            "3": 0.3,
            "4": 0.2,
            "5": 0.1,
        },
        ligand_fixed_size_bfs_prob: float = 0.9,
        supplementary_ligand_distance: float = 5.0,
        supplementary_ligand_prob: float = 0.5,
        **kwargs,
    ):
        """
        Args:
            ligand_motif_prob (float): Probability _generate_ligand_motif_mask is used for ligand complexes.
            pocket_motif_prob (float): Probability _generate_pocket_motif_mask is used for complexes with a pocket.
            whole_ligand_prob (float): Probability the whole ligand is used in the motif.
            drop_ligand_motif_prob (float): Probability of not including ligand atoms in the motif generated
                around the ligand.
            whole_sidechain_prob (float): Probability the whole sidechain is included vs partial.
            pocket_score_threshold (float): Threshold for pocket score to identify seed atoms.
            pocket_n_neighbours (int): Number of neighbours for pocket score calculation.
            sidechain_seed_is_tip_prob (float): Probability tip atom is used as seed.
            sidechain_seed_is_in_sphere_prob (float): Probability in-sphere atom is used as seed.
            sidechain_bfs_depth_prob (float): Probability parameter for geometric distribution of BFS depth.
            residue_inclusion_prob (float): Probability of including a residue in sphere.
            r_min (float): Minimum radius in angstroms for sphere search.
            r_max (float): Maximum radius in angstroms for sphere search.
            max_motif_residues (int): Maximum number of residues allowed in motif.
            motif_train_rate (float): Training rate for motif generation.
            use_tip_atoms_only_prob (float): Probability of using only tip atoms in the motif.
            complex_ligand_threshold (int): Minimum atom count to classify a ligand as "complex".
                Ligands with fewer atoms are considered "simple" (e.g., metal ions, small molecules).
            complex_ligand_preference_prob (float): Probability of selecting from complex ligands
                (if any exist) rather than sampling uniformly from all ligands.
            ligand_region_count_probs (dict[str, float]): Mapping from number of regions (as string)
                to probability. Defaults to {"1": 0.5, "2": 0.4, "3": 0.07, "4": 0.03}.
            ligand_bfs_atom_count_probs (dict[str, float]): Mapping from number of atoms per BFS
                region (as string) to probability. Defaults to {"1": 0.15, ..., "5": 0.1}.
            ligand_fixed_size_bfs_prob (float): Probability of using fixed-size BFS (sampling from
                ligand_bfs_atom_count_probs) vs proportion-based BFS. Defaults to 0.9.
            supplementary_ligand_distance (float): Maximum distance (in Angstroms) from seed point
                to consider nearby ligand atoms for supplementary ligand inclusion. Defaults to 5.0.
            supplementary_ligand_prob (float): Probability of including each detected nearby ligand
                chain as a supplementary ligand in the motif. Defaults to 0.5.
        """
        self.ligand_motif_prob = ligand_motif_prob
        self.pocket_motif_prob = pocket_motif_prob
        self.whole_ligand_prob = whole_ligand_prob
        self.drop_ligand_motif_prob = drop_ligand_motif_prob
        self.whole_sidechain_prob = whole_sidechain_prob
        self.pocket_score_threshold = pocket_score_threshold
        self.pocket_n_neighbours = pocket_n_neighbours
        self.sidechain_seed_is_tip_prob = sidechain_seed_is_tip_prob
        self.sidechain_seed_is_in_sphere_prob = sidechain_seed_is_in_sphere_prob
        self.sidechain_bfs_depth_prob = sidechain_bfs_depth_prob
        self.residue_inclusion_prob = residue_inclusion_prob
        self.use_tip_atoms_only_prob = use_tip_atoms_only_prob
        self.r_min = r_min
        self.r_max = r_max
        self.max_motif_residues = max_motif_residues
        self.complex_ligand_threshold = complex_ligand_threshold
        self.complex_ligand_preference_prob = complex_ligand_preference_prob
        self.ligand_region_count_probs = ligand_region_count_probs
        self.ligand_bfs_atom_count_probs = ligand_bfs_atom_count_probs
        self.ligand_fixed_size_bfs_prob = ligand_fixed_size_bfs_prob
        self.supplementary_ligand_distance = supplementary_ligand_distance
        self.supplementary_ligand_prob = supplementary_ligand_prob

    def make_motif(self, atom_array: AtomArray) -> AtomArray:
        """
        Annotates the atom array with is_motif attribute. This is the main public function.

        Args:
            atom_array (AtomArray): Input Biotite AtomArray

        Returns:
            AtomArray: The same atom array with is_motif annotation added (int mask)
        """
        # TODO: delete this check once we have pocket annotations for the PDB
        has_pocket_id = "pocket_id" in atom_array.get_annotation_categories()

        # Generate motif mask
        if np.random.rand() < self.ligand_motif_prob and atom_array.is_ligand.any():
            is_motif_mask = self._generate_ligand_motif_mask(atom_array)
        elif (
            has_pocket_id
            and np.random.rand() < self.pocket_motif_prob
            and atom_array.pocket_id.any()
        ):
            is_motif_mask = self._generate_pocket_motif_mask(atom_array)
        else:
            is_motif_mask = self._generate_chain_motif_mask(atom_array)

        # Validate motif size
        if is_motif_mask.sum() == 1:
            # TODO: we should tweak things to make this less frequent.
            # logger.warning("Motif only has one atom. Setting all is_motif to 0.")
            is_motif_mask = np.zeros(len(atom_array), dtype=int)
        elif is_motif_mask.any():
            num_residues_in_motif = len(np.unique(atom_array[is_motif_mask > 0].res_id))
            if num_residues_in_motif > self.max_motif_residues:
                # logger.warning(
                #     f"Motif has too many residues: {num_residues_in_motif}. "
                #     f"Setting all is_motif to 0."
                # )
                is_motif_mask = np.zeros(len(atom_array), dtype=int)

        atom_array.set_annotation("is_motif", is_motif_mask)

        is_motif_related = self._get_is_motif_related_mask(atom_array)
        atom_array.set_annotation("is_motif_related", is_motif_related)

        return atom_array

    def _get_is_motif_related_mask(self, atom_array: AtomArray) -> np.ndarray:
        """
        Generates a mask indicating which atoms are related to the motif. This includes the motif
        atoms themselves and the rest of the atoms which lie in a residue (determined by ref_space_uid)
        that contains at least one motif atom.

        Args:
            atom_array (AtomArray): Input Biotite AtomArray

        Returns:
            np.ndarray: Integer mask (0/1) indicating which atoms are related to the motif
        """
        is_motif_related = np.zeros(len(atom_array), dtype=int)

        # Get unique residues in motif
        unique_ref_space_uids = np.unique(
            atom_array[atom_array.is_motif > 0].ref_space_uid
        )

        # Mark all atoms in these residues as related to the motif
        is_motif_related[np.isin(atom_array.ref_space_uid, unique_ref_space_uids)] = 1

        return is_motif_related

    def _sample_bfs_atom_count(self, ligand_graph_size: int) -> int:
        """
        Sample the number of atoms to collect via BFS for a ligand region.

        With probability ligand_fixed_size_bfs_prob, samples from ligand_bfs_atom_count_probs.
        Otherwise, uses proportion-based sampling.

        Args:
            ligand_graph_size (int): Number of nodes in the ligand graph.

        Returns:
            int: Number of atoms to collect via BFS.
        """
        if np.random.rand() < self.ligand_fixed_size_bfs_prob:
            atom_counts = [int(k) for k in self.ligand_bfs_atom_count_probs.keys()]
            atom_probs = list(self.ligand_bfs_atom_count_probs.values())
            return int(np.random.choice(atom_counts, p=atom_probs))
        else:
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
        within_distance_mask = distances <= self.supplementary_ligand_distance
        if not within_distance_mask.any():
            return supplementary_ligand_mask

        # Get unique ligand chains that have atoms within distance
        nearby_indices = other_ligand_indices[within_distance_mask]
        nearby_distances = distances[within_distance_mask]
        nearby_asym_ids = atom_array.asym_id_int[nearby_indices]
        unique_nearby_asym_ids = np.unique(nearby_asym_ids)

        # Process each nearby ligand chain independently
        for asym_id in unique_nearby_asym_ids:
            if np.random.rand() >= self.supplementary_ligand_prob:
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
            if atom_count >= self.complex_ligand_threshold:
                complex_ligand_ids.append(asym_id)

        # Prefer complex ligands if available
        if (
            len(complex_ligand_ids) > 0
            and np.random.rand() < self.complex_ligand_preference_prob
        ):
            ligand_motif_asym_id = np.random.choice(complex_ligand_ids)
        else:
            ligand_motif_asym_id = np.random.choice(ligand_asym_ids)

        ligand_motif_chain_mask = (
            atom_array.asym_id_int == ligand_motif_asym_id
        ) & ligand_mask

        if (
            np.random.rand() < self.whole_ligand_prob
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
                # logger.warning("No nodes in ligand graph. Returning empty motif mask.")
                # logger.warning(
                #     f"Number of ligand atoms involved in this empty graph: {len(ligand_array)}"
                # )
                return is_motif

            region_counts = [int(k) for k in self.ligand_region_count_probs.keys()]
            region_probs = list(self.ligand_region_count_probs.values())
            num_regions = int(np.random.choice(region_counts, p=region_probs))

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

        if np.random.rand() < self.drop_ligand_motif_prob:
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
        Generates a motif mask centered around a pocket on a polymer chain.

        Args:
            atom_array (AtomArray): Input Biotite AtomArray

        Returns:
            np.ndarray: Integer mask (0/1) indicating motif atoms
        """
        is_motif = np.zeros(len(atom_array), dtype=int)

        # Filter for polymer chains
        polymer_mask = np.isin(atom_array.chain_mol_type, ["protein", "RNA", "DNA"])

        if not polymer_mask.any():
            # logger.warning(
            #     "No polymer chains in the atom array. Setting all is_motif to 0."
            # )
            return is_motif

        # Pick a random polymer chain
        polymer_atoms = atom_array[polymer_mask]
        chain_id = np.random.choice(np.unique(polymer_atoms.asym_id_int))

        chain_mask = atom_array.asym_id_int == chain_id
        chain_array = atom_array[chain_mask]

        # Compute pocket scores
        pocket_scores = compute_pocket_scores(
            chain_array, n_neighbors=self.pocket_n_neighbours
        )

        # Identify pocket atoms in chain
        pocket_atom_indices_in_chain = np.where(
            pocket_scores <= self.pocket_score_threshold
        )[0]

        if len(pocket_atom_indices_in_chain) == 0:
            # logger.warning("No pocket atoms found. Setting all is_motif to 0.")
            return is_motif

        # Get seed point from pocket
        pocket_atoms = chain_array[pocket_atom_indices_in_chain]
        seed_coord = get_random_seed_atom_from_array(pocket_atoms)

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
        if np.random.rand() < self.use_tip_atoms_only_prob:
            use_tip_atoms_only = True
        else:
            use_tip_atoms_only = False

        is_motif = np.zeros(len(atom_array), dtype=int)

        # Find polymer atoms within sphere
        r = np.random.uniform(self.r_min, self.r_max)

        distances = np.linalg.norm(atom_array.coord - point_coord, axis=-1)
        within_radius_mask = distances <= r

        polymer_mask = np.isin(atom_array.chain_mol_type, ["protein", "RNA", "DNA"])
        atoms_in_sphere_mask = within_radius_mask & polymer_mask

        # Apply optional mask
        if optional_mask is not None:
            atoms_in_sphere_mask = atoms_in_sphere_mask & optional_mask

        if not atoms_in_sphere_mask.any():
            # logger.warning(
            #     "No polymer atoms close to seed point. Setting all is_motif to 0."
            # )
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
            # logger.warning("No residues in sphere. Setting all is_motif to 0.")
            return is_motif

        # Process each residue
        for uid in unique_ref_space_uids:
            # Skip residue with probability
            if np.random.rand() > self.residue_inclusion_prob:
                continue

            # Get all atoms of this residue
            residue_mask = atom_array.ref_space_uid == uid

            if np.random.rand() > self.whole_sidechain_prob:
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
            # logger.warning("No nodes in residue graph. Setting all is_motif to 0.")
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

        if p < self.sidechain_seed_is_in_sphere_prob and valid_seed_atoms:
            seed_atom = np.random.choice(valid_seed_atoms)
        elif (
            p < self.sidechain_seed_is_in_sphere_prob + self.sidechain_seed_is_tip_prob
        ):
            seed_atom = graph_nodes[
                -1
            ]  # TODO: I don't think the last node in the graph is neccessarily the tip
        else:
            seed_atom = np.random.choice(graph_nodes)

        # Pick depth
        num_bonds = np.random.geometric(self.sidechain_bfs_depth_prob) - 1

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


def compute_pocket_scores(chain_array: AtomArray, n_neighbors: int = 25) -> np.ndarray:
    """
    Compute pocket scores for atoms in the array. The score is the distance between the target
    atom and the centroid of the nearest n_neighbors neighbours.

    Args:
        chain_array (AtomArray): AtomArray containing all atoms
        n_neighbors (int): Number of nearest neighbors to use for centroid calculation

    Returns:
        np.ndarray: Array of pocket scores for atoms. Lower scores indicate deeper pockets.

    Notes:
        - Lower pocket_score means the atom is more likely in a pocket
        - Typical pocket threshold: score < 0.18 Å suggests pocket atom (we think)
        - Adjust n_neighbors based on protein size (20-30 works well for most cases)
    """
    # Query for n_neighbors
    coords = chain_array.coord
    tree = KDTree(coords)
    _, indices = tree.query(coords, k=min(len(coords), n_neighbors + 1))
    neighbor_indices = indices[:, 1:]

    # Calculate pocket scores for each atom
    scores = np.zeros(len(coords))
    for i, neighbor_idx in enumerate(neighbor_indices):
        neighbor_coords = coords[neighbor_idx]
        centroid = np.mean(neighbor_coords, axis=0)

        # Distance from point to centroid (pocket score)
        scores[i] = np.linalg.norm(coords[i] - centroid)
    return scores


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
