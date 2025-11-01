"""
Houses all the logic for generating motifs during model training.
"""

from logging import getLogger

import networkx as nx
import numpy as np
from biotite.structure import AtomArray

from protenix.data.motif_utils import (
    bfs_nodes_by_count,
    compute_pocket_scores,
    get_random_seed_atom_from_array,
)

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
        whole_ligand_prob: float = 0.7,
        whole_sidechain_prob: float = 0.2,
        pocket_score_threshold: float = 0.18,
        pocket_n_neighbours: int = 25,
        sidechain_seed_is_tip_prob: float = 0.8,
        sidechain_bfs_depth_prob: float = 0.5,
        residue_inclusion_prob: float = 0.9,
        r_min: float = 3.0,
        r_max: float = 6.0,
        max_motif_residues: int = 20,
        motif_train_rate: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            ligand_motif_prob (float): Probability _generate_ligand_motif_mask is used for ligand complexes.
            whole_ligand_prob (float): Probability the whole ligand is used in the motif.
            whole_sidechain_prob (float): Probability the whole sidechain is included vs partial.
            pocket_score_threshold (float): Threshold for pocket score to identify seed atoms.
            pocket_n_neighbours (int): Number of neighbours for pocket score calculation.
            sidechain_seed_is_tip_prob (float): Probability tip atom is used as seed.
            sidechain_bfs_depth_prob (float): Probability parameter for geometric distribution of BFS depth.
            residue_inclusion_prob (float): Probability of including a residue in sphere.
            r_min (float): Minimum radius in angstroms for sphere search.
            r_max (float): Maximum radius in angstroms for sphere search.
            max_motif_residues (int): Maximum number of residues allowed in motif.
            motif_train_rate (float): Training rate for motif generation.
        """
        self.ligand_motif_prob = ligand_motif_prob
        self.whole_ligand_prob = whole_ligand_prob
        self.whole_sidechain_prob = whole_sidechain_prob
        self.pocket_score_threshold = pocket_score_threshold
        self.pocket_n_neighbours = pocket_n_neighbours
        self.sidechain_seed_is_tip_prob = sidechain_seed_is_tip_prob
        self.sidechain_bfs_depth_prob = sidechain_bfs_depth_prob
        self.residue_inclusion_prob = residue_inclusion_prob
        self.r_min = r_min
        self.r_max = r_max
        self.max_motif_residues = max_motif_residues

    def make_motif(self, atom_array: AtomArray) -> AtomArray:
        """
        Annotates the atom array with is_motif attribute. This is the main public function.

        Args:
            atom_array (AtomArray): Input Biotite AtomArray

        Returns:
            AtomArray: The same atom array with is_motif annotation added (int mask)
        """
        # Generate motif mask
        if atom_array.is_ligand.any() and np.random.rand() < self.ligand_motif_prob:
            is_motif_mask = self._generate_ligand_motif_mask(atom_array)
        else:
            is_motif_mask = self._generate_chain_motif_mask(atom_array)

        # Validate motif size
        if is_motif_mask is not None and is_motif_mask.any():
            num_residues_in_motif = len(np.unique(atom_array[is_motif_mask > 0].res_id))
            if num_residues_in_motif > self.max_motif_residues:
                logger.warning(
                    f"Motif has too many residues: {num_residues_in_motif}. "
                    f"Setting all is_motif to 0."
                )
                is_motif_mask = np.zeros(len(atom_array), dtype=int)
        else:
            # No motif generated
            is_motif_mask = np.zeros(len(atom_array), dtype=int)

        # Add is_motif annotation to atom array
        atom_array.set_annotation("is_motif", is_motif_mask)

        # Add is_motif_related annotation to atom array
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

    def _generate_ligand_motif_mask(self, atom_array: AtomArray) -> np.ndarray:
        """
        Generates a motif mask centered around a ligand.

        Args:
            atom_array (AtomArray): Input Biotite AtomArray

        Returns:
            np.ndarray: Integer mask (0/1) indicating motif atoms
        """
        is_motif = np.zeros(len(atom_array), dtype=int)

        # Select a random ligand
        ligand_mask = atom_array.chain_mol_type == "ligand"
        ligand_asym_ids = np.unique(atom_array.label_asym_id[ligand_mask])
        ligand_motif_asym_id = np.random.choice(ligand_asym_ids)

        ligand_atom_mask = (
            atom_array.label_asym_id == ligand_motif_asym_id
        ) & ligand_mask

        if np.random.rand() < self.whole_ligand_prob or ligand_atom_mask.sum() == 1:
            # Add whole ligand to motif
            is_motif[ligand_atom_mask] = 1

            # Get seed point from ligand
            ligand_atoms = atom_array[ligand_atom_mask]
            seed_coord = get_random_seed_atom_from_array(ligand_atoms)

            # Add surrounding residues
            surrounding_mask = self._generate_motif_around_point_mask(
                point_coord=seed_coord,
                atom_array=atom_array,
            )
            is_motif[surrounding_mask > 0] = 1

        else:
            # Add partial ligand to motif
            ligand_array = atom_array[ligand_atom_mask]
            ligand_graph = ligand_array.bonds.as_graph()
            graph_nodes = list(ligand_graph.nodes())

            if len(graph_nodes) == 0:
                logger.warning("No nodes in ligand graph. Returning empty motif mask.")
                return is_motif

            seed_ligand = np.random.choice(graph_nodes)
            ligand_proportion = np.random.rand()
            num_lig_atoms_in_motif = max(1, int(ligand_proportion * len(ligand_graph)))

            # Get nodes via BFS
            nodes = bfs_nodes_by_count(
                ligand_graph, seed_ligand, num_lig_atoms_in_motif
            )

            # Mark these ligand atoms as motif
            full_ligand_indices = np.where(ligand_atom_mask)[0]
            partial_ligand_indices = full_ligand_indices[nodes]
            is_motif[partial_ligand_indices] = 1

            # Get seed point from partial ligand
            partial_ligand_atoms = atom_array[partial_ligand_indices]
            seed_coord = get_random_seed_atom_from_array(partial_ligand_atoms)

            # Add surrounding residues
            surrounding_mask = self._generate_motif_around_point_mask(
                point_coord=seed_coord,
                atom_array=atom_array,
            )
            is_motif[surrounding_mask > 0] = 1

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
            logger.warning(
                "No polymer chains in the atom array. Returning empty motif mask."
            )
            return is_motif

        # Pick a random polymer chain
        polymer_atoms = atom_array[polymer_mask]
        chain_id = np.random.choice(np.unique(polymer_atoms.label_asym_id))

        chain_mask = atom_array.label_asym_id == chain_id
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
            logger.warning("No pocket atoms found. Returning empty motif mask.")
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
    ) -> np.ndarray:
        """
        Generates a motif mask around a point by identifying residues within a sphere.

        Args:
            point_coord (np.ndarray): Center point coordinates
            atom_array (AtomArray): AtomArray to generate motif from

        Returns:
            np.ndarray: Integer mask (0/1) indicating motif atoms
        """
        is_motif = np.zeros(len(atom_array), dtype=int)

        # Pick radius
        r = np.random.uniform(self.r_min, self.r_max)

        # Calculate distances
        distances = np.linalg.norm(atom_array.coord - point_coord, axis=-1)
        within_radius_mask = distances <= r

        # Filter for polymer atoms in sphere
        polymer_mask = np.isin(atom_array.chain_mol_type, ["protein", "RNA", "DNA"])
        atoms_in_sphere_mask = within_radius_mask & polymer_mask

        if not atoms_in_sphere_mask.any():
            logger.warning(
                "No polymer atoms close to seed point. Returning empty mask."
            )
            return is_motif

        # Get unique residues in sphere
        atoms_in_sphere = atom_array[atoms_in_sphere_mask]
        unique_ref_space_uids = np.unique(atoms_in_sphere.ref_space_uid)

        if len(unique_ref_space_uids) == 0:
            logger.warning("No residues in sphere. Returning empty mask.")
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
                    atom_array, residue_mask
                )
                is_motif[partial_mask > 0] = 1
            else:
                # Whole residue
                is_motif[residue_mask > 0] = 1

        return is_motif

    def _get_partial_residue_motif_mask(
        self, atom_array: AtomArray, residue_mask: np.ndarray
    ) -> np.ndarray:
        """
        Gets partial residue motif mask using BFS on the residue bond graph.
        TODO: I need to double check if this method even works at all. Some minor evidence
        suggests that we are just getting the warning every time (i.e do we ever even have the bond information?)

        Args:
            atom_array (AtomArray): Full atom array
            residue_mask (np.ndarray): Boolean mask indicating which atoms belong to the residue

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
            logger.warning("No nodes in residue graph. Returning empty motif mask.")
            return partial_motif_mask

        # Pick seed atom
        if np.random.rand() < self.sidechain_seed_is_tip_prob:
            tip_atom = graph_nodes[-1]  # TODO: is the tip atom even the last node?
        else:
            tip_atom = np.random.choice(graph_nodes)

        # Pick depth
        num_bonds = np.random.geometric(self.sidechain_bfs_depth_prob) - 1

        # Get atoms within num_bonds of seed
        motif_atom_ids = nx.single_source_shortest_path_length(
            G_res, tip_atom, cutoff=num_bonds
        )
        motif_atom_ids = list(motif_atom_ids.keys())

        # Map back to full array
        residue_indices = np.where(residue_mask)[0]
        selected_indices = residue_indices[motif_atom_ids]
        partial_motif_mask[selected_indices] = 1

        return partial_motif_mask
