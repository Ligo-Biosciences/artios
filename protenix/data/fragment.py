"""
Houses all the logic for generating fragments from protein structures.
"""

from logging import getLogger

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import KDTree

from protenix.data.pocket import find_confident_pockets

logger = getLogger(__name__)


class FragmentMaker:
    """
    Main class for generating fragments when prepping the dataset in parser.py.
    The AtomArray of a .cif file is handed to the `FragmentMaker`, which will then
    remove disordered residues (according to pLDDT). The remaining 'fragmented'
    structure undergoes fragment specfic processing, filtering fragments that do
    not favour the training-relevant criteria (i.e for motif scaffolding, fragments
    with a high `Rg` are filtered out). The exact workflow is as follows:

    - filter_low_plddt_residues
        Filters the residues of the input AtomArray according to residue-level
        pLDDT (from the .b_factor annotation). Returns a (modified) AtomArray.
        The pLDDT threshold is class attribute.
    - assign_fragment_ids
        Detects fragments in the AtomArray left by the filtering. This detects the contiguous
        windows of residues. Each window gets a unique fragment ID. Fragment ID is assigned
        using an AtomArray annotation (starting from 0).
    - spatially_cluster_fragments
        This identifies which fragments are
        Fragments that are 'in contact' are assigned the same fragment cluster ID.
    - calculate_fragment_metrics
        This calculates the avg pLDDT, length and Rg of each fragment. This is stored in a dictionary
        of fragment_id -> metric.
    - filter_fragments
        This filters the fragments based on the calculated metrics as well as whether or not the fragment
        contains a pocket (using a .pocket annotation the AtomArray has). Filter values are class attributes.
        Fragments that do not meet the criteria are filtered out.
    - correct_fragment_ids
        This corrects the remaining fragment cluster ID's to ensure that they start from 0 and are contiguous.
        This is required as filtering fragments will result in gaps in the fragment ID's.
    """

    def __init__(
        self,
        plddt_disorder_threshold: float = 50,
        contact_distance_threshold: float = 4.0,
        min_contact_residues: int = 4,
        min_fragment_length: int = 150,
        min_fragment_plddt: float = 70,
    ):
        """
        Initialize FragmentMaker with filtering and clustering thresholds.

        Args:
            plddt_disorder_threshold (float): Threshold for pLDDT to filter residues. Defaults to 50.
            contact_distance_threshold (float): Distance threshold for atom-atom contacts in spatial clustering. Defaults to 4.0 Angstroms.
            min_contact_residues (int): Minimum number of residues with contacts to merge fragments. Defaults to 4.
            min_fragment_length (int): Minimum length of a fragment. Defaults to 150.
            min_fragment_plddt (float): Minimum pLDDT of a fragment. Defaults to 70.
        """
        self.plddt_residue_threshold = plddt_disorder_threshold
        self.contact_distance_threshold = contact_distance_threshold
        self.min_contact_residues = min_contact_residues
        self.min_fragment_length = min_fragment_length
        self.min_fragment_plddt = min_fragment_plddt

        # Create length-dependent Rg upper bound
        self.rg_upper_bound_line = (
            0.03457943925233645,
            12.813084112149532,
        )  # slope and y-intercept w.r.t length

    def filter_low_plddt_residues(self, atom_array: AtomArray) -> AtomArray:
        """
        Filter residues based on pLDDT threshold and initialize fragment_id annotation.

        Residues with mean pLDDT below threshold are marked with fragment_id = -1.
        All other residues are initialized with fragment_id = 0 (to be reassigned later).

        Args:
            atom_array (AtomArray): Biotite AtomArray object with b_factor annotation.

        Returns:
            AtomArray: AtomArray with fragment_id annotation added.
        """
        # Initialize all atoms with fragment_id = 0
        fragment_ids = np.zeros(len(atom_array), dtype=np.int32)

        # Mark low pLDDT residues with -1 (vectorized operation)
        low_plddt_mask = atom_array.b_factor < self.plddt_residue_threshold
        fragment_ids[low_plddt_mask] = -1

        atom_array.set_annotation("fragment_id", fragment_ids)
        return atom_array

    def assign_fragment_ids(self, atom_array: AtomArray) -> AtomArray:
        """
        Assign unique fragment IDs to contiguous windows of residues.

        Consecutive residues (by res_id) within the same chain that are not marked
        as -1 are assigned the same fragment_id. fragment_id starts from 1.

        Args:
            atom_array (AtomArray): AtomArray with fragment_id annotation initialized.

        Returns:
            AtomArray: AtomArray with fragment_id annotation updated.
        """
        fragment_ids = atom_array.fragment_id.copy()
        current_fragment_id = 0

        res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        prev_res_id = None
        prev_fragment_valid = False

        for start, stop in zip(res_starts[:-1], res_starts[1:]):
            curr_res_id = atom_array.res_id[start]
            curr_fragment_valid = fragment_ids[start] != -1

            if not curr_fragment_valid:
                # Skip residues marked as -1
                prev_res_id = curr_res_id
                prev_fragment_valid = False
                continue

            # Check if this is contiguous with previous residue
            is_contiguous = (
                prev_fragment_valid
                and prev_res_id is not None
                and curr_res_id - prev_res_id == 1
            )

            if is_contiguous:
                # Continue current fragment
                fragment_ids[start:stop] = current_fragment_id
            else:
                # Start new fragment
                current_fragment_id += 1
                fragment_ids[start:stop] = current_fragment_id

            prev_res_id = curr_res_id
            prev_fragment_valid = True

        atom_array.set_annotation("fragment_id", fragment_ids)
        return atom_array

    def _have_sufficient_contacts(
        self, atom_array: AtomArray, frag_id_i: int, frag_id_j: int
    ) -> bool:
        """
        Check if two fragments have sufficient spatial contacts.

        Returns True if at least min_contact_residues residues from frag_id_i have atoms
        within contact_distance_threshold of any atom in frag_id_j.

        Args:
            atom_array (AtomArray): AtomArray with fragment_id annotation.
            frag_id_i (int): First fragment ID.
            frag_id_j (int): Second fragment ID.

        Returns:
            bool: True if fragments have >= min_contact_residues residues with close contacts.
        """
        # Get atoms for each fragment
        atoms_i = atom_array[atom_array.fragment_id == frag_id_i]
        atoms_j = atom_array[atom_array.fragment_id == frag_id_j]

        # Find atoms within contact distance
        tree_i = KDTree(atoms_i.coord)
        tree_j = KDTree(atoms_j.coord)
        close_pairs = tree_i.query_ball_tree(tree_j, r=self.contact_distance_threshold)

        # Find atoms from frag_i that have close contacts
        has_contact = np.array([len(pairs) > 0 for pairs in close_pairs])

        # Count unique residues with contacts
        res_ids_with_contact = atoms_i.res_id[has_contact]
        unique_res_count = len(np.unique(res_ids_with_contact))

        return unique_res_count >= self.min_contact_residues

    def spatially_cluster_fragments(self, atom_array: AtomArray) -> AtomArray:
        """
        Cluster spatially close fragments by assigning fragment_cluster_id.

        Two fragments are in the same cluster if at least min_contact_residues residues
        from one fragment have atoms within contact_distance_threshold of any atom in the
        other fragment. Uses Union-Find algorithm to ensure transitivity: if A is close
        to B and B is close to C, then A and C get the same fragment_cluster_id.

        Args:
            atom_array (AtomArray): AtomArray with fragment_id annotation.

        Returns:
            AtomArray: AtomArray with fragment_cluster_id annotation added.
        """
        # Initialize fragment_cluster_id from fragment_id
        fragment_cluster_id = atom_array.fragment_id.copy()

        # Get unique fragment IDs (excluding -1)
        unique_frag_ids = np.unique(atom_array.fragment_id)
        unique_frag_ids = unique_frag_ids[unique_frag_ids != -1]

        if len(unique_frag_ids) == 0:
            atom_array.set_annotation("fragment_cluster_id", fragment_cluster_id)
            return atom_array

        # Union-Find data structure
        parent = {frag_id: frag_id for frag_id in unique_frag_ids}

        def find(x):
            """Find root parent with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            """Union two sets."""
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        # Find fragment pairs with sufficient contacts and union them
        frag_ids_list = list(unique_frag_ids)
        for i in range(len(frag_ids_list)):
            for j in range(i + 1, len(frag_ids_list)):
                frag_i = frag_ids_list[i]
                frag_j = frag_ids_list[j]

                if self._have_sufficient_contacts(atom_array, frag_i, frag_j):
                    union(frag_i, frag_j)

        # Remap fragment IDs to their root parent (cluster ID)
        for frag_id in unique_frag_ids:
            root = find(frag_id)
            mask = atom_array.fragment_id == frag_id
            fragment_cluster_id[mask] = root

        # Replace fragment id annotation with cluster ids
        atom_array.del_annotation("fragment_id")
        atom_array.set_annotation("fragment_id", fragment_cluster_id)
        return atom_array

    def calculate_fragment_metrics(self, atom_array: AtomArray) -> dict:
        """
        Calculate metrics for each fragment: average pLDDT, length, and radius of gyration.

        Args:
            atom_array (AtomArray): AtomArray with fragment_id annotation.

        Returns:
            dict: Dictionary mapping fragment_id to metrics dict with keys:
                - avg_plddt (float): Average pLDDT of fragment
                - length (int): Number of unique residues in fragment
                - rg (float): Radius of gyration in Angstroms
        """
        metrics = {}

        # Get unique fragment IDs (excluding -1)
        unique_frag_ids = np.unique(atom_array.fragment_id)
        unique_frag_ids = unique_frag_ids[unique_frag_ids != -1]

        for frag_id in unique_frag_ids:
            mask = atom_array.fragment_id == frag_id
            frag_atoms = atom_array[mask]

            # Calculate average pLDDT
            avg_plddt = np.mean(frag_atoms.b_factor)

            # Calculate length (number of unique residues)
            unique_res_ids = np.unique(frag_atoms.res_id)
            length = len(unique_res_ids)

            # Calculate radius of gyration
            centroid = np.mean(frag_atoms.coord, axis=0)
            squared_distances = np.sum((frag_atoms.coord - centroid) ** 2, axis=1)
            rg = np.sqrt(np.mean(squared_distances))

            # Calculate mean atom SASA
            atom_sasa = struc.sasa(frag_atoms, vdw_radii="Single")
            sasa = np.nanmean(atom_sasa)

            metrics[frag_id] = {
                "avg_plddt": avg_plddt,
                "length": length,
                "rg": rg,
                "sasa": sasa,
            }

        return metrics

    def find_pockets_in_fragments(self, atom_array: AtomArray) -> dict:
        """
        Finds pockets in fragments using Pocketeer.

        Args:
            atom_array (AtomArray): AtomArray with fragment_id annotation.

        Returns:
            dictionary mapping fragment  id to lists of per-pocket statistics.
        """
        fragment_pockets = {}

        # Get unique fragment IDs (excluding -1)
        unique_frag_ids = np.unique(atom_array.fragment_id)
        unique_frag_ids = unique_frag_ids[unique_frag_ids != -1]

        # Add atom id field for proper indexing
        atom_id = np.arange(len(atom_array))
        atom_array.set_annotation("atom_id", atom_id)

        for frag_id in unique_frag_ids:
            mask = atom_array.fragment_id == frag_id
            frag_atoms = atom_array[mask]
            pockets = find_confident_pockets(
                frag_atoms,
                plddt_cutoff=70.0,
                polar_probe_radius=5.0,
                min_spheres=35,
                sasa_cutoff=10.0,
                merge_distance=1.75,
            )
            tree = KDTree(frag_atoms.coord)
            centroids = []
            volumes = []
            pocket_atoms = []
            r15_atom_counts = []
            for pocket in pockets:
                centroids.append(pocket.centroid)
                volumes.append(pocket.volume)

                pocket_atom_indices = set()
                for sphere in pocket.spheres:
                    pocket_atom_indices.update(sphere.atom_indices)

                # Get the global index from local index for these pocket atoms
                pocket_atom_indices = list(pocket_atom_indices)
                global_atom_ids = frag_atoms.atom_id[pocket_atom_indices]
                pocket_atoms.append(global_atom_ids.tolist())

                # Count atoms within 15 Angstrom radius of pocket centroid
                r15_indices = tree.query_ball_point(pocket.centroid, r=15.0)
                r15_atom_counts.append(len(r15_indices))

            fragment_pockets[frag_id] = {
                "centroids": centroids,
                "radius_15_atom_counts": r15_atom_counts,
                "pocket_atoms": pocket_atoms,
                "volumes": volumes,
            }

        return fragment_pockets

    def filter_fragments(self, atom_array: AtomArray) -> AtomArray:
        """
        Filter fragments based on calculated metrics.

        Fragments that do not meet all criteria are marked with fragment_id = -1.
        Criteria:
        - length >= min_fragment_length
        - avg_plddt >= min_fragment_plddt
        - rg <= length-dependent upper bound (linear equation: slope * length + y_intercept)

        Args:
            atom_array (AtomArray): AtomArray with fragment_id annotation.

        Returns:
            AtomArray: AtomArray with filtered fragments marked as -1.
        """
        fragment_ids = atom_array.fragment_id.copy()
        metrics = self.calculate_fragment_metrics(atom_array)

        for frag_id, frag_metrics in metrics.items():
            # Check if fragment meets all criteria
            passes_length = frag_metrics["length"] >= self.min_fragment_length
            passes_plddt = frag_metrics["avg_plddt"] >= self.min_fragment_plddt

            # Calculate length-dependent Rg upper bound using linear equation
            slope, y_intercept = self.rg_upper_bound_line
            max_rg = slope * frag_metrics["length"] + y_intercept
            passes_rg = frag_metrics["rg"] <= max_rg

            if not (passes_length and passes_plddt and passes_rg):
                # Mark this fragment as filtered
                mask = atom_array.fragment_id == frag_id
                fragment_ids[mask] = -1

        atom_array.set_annotation("fragment_id", fragment_ids)
        return atom_array

    def get_fragment_annotation(self, atom_array: AtomArray) -> AtomArray:
        """
        Calculate and assign fragment_id annotation to the input AtomArray.

        This orchestrates the complete workflow:
        1. Filter low pLDDT residues
        2. Assign initial fragment IDs to contiguous regions
        3. Spatially cluster fragments
        4. Calculate fragment metrics
        5. Filter fragments based on metrics
        6. Correct fragment IDs to be contiguous

        Args:
            atom_array (AtomArray): Input AtomArray with b_factor annotation.

        Returns:
            AtomArray: AtomArray with fragment_id annotation added.
                fragment_id = -1 for filtered atoms, otherwise 0, 1, 2, ...
        """
        atom_array = self.filter_low_plddt_residues(atom_array)
        atom_array = self.assign_fragment_ids(atom_array)
        atom_array = self.spatially_cluster_fragments(atom_array)
        atom_array = self.filter_fragments(atom_array)
        return atom_array

    def get_fragment_statistics(self, atom_array: AtomArray) -> dict:
        """
        Calculate metrics per-fragment after spatial clustering and return a
        dictionary of lists that includes the metrics per-fragment as well as all
        the pockets in different fragments.

        Args:
            atom_array (AtomArray): Input AtomArray

        Returns:
            a dictionary with per-fragment metrics and pocket data.
        """
        atom_array = self.filter_low_plddt_residues(atom_array)
        atom_array = self.assign_fragment_ids(atom_array)
        atom_array = self.spatially_cluster_fragments(atom_array)

        metrics = self.calculate_fragment_metrics(atom_array)
        pockets = self.find_pockets_in_fragments(atom_array)

        fragment_ids = sorted(metrics.keys())

        # Organize fragment metrics and pocket data by metric name
        metrics_by_name = {
            # Fragment-level metrics
            "fragment_avg_plddt": [
                metrics[frag_id]["avg_plddt"] for frag_id in fragment_ids
            ],
            "fragment_length": [metrics[frag_id]["length"] for frag_id in fragment_ids],
            "fragment_rg": [metrics[frag_id]["rg"] for frag_id in fragment_ids],
            "fragment_sasa": [metrics[frag_id]["sasa"] for frag_id in fragment_ids],
            # Pocket-level metrics
            "fragment_pocket_centroids": [
                pockets[frag_id].get("centroids", []) for frag_id in fragment_ids
            ],
            "fragment_pocket_radius_15_atom_counts": [
                pockets[frag_id].get("radius_15_atom_counts", [])
                for frag_id in fragment_ids
            ],
            "fragment_pocket_atoms": [
                pockets[frag_id].get("pocket_atoms", []) for frag_id in fragment_ids
            ],
            "fragment_pocket_volumes": [
                pockets[frag_id].get("volumes", []) for frag_id in fragment_ids
            ],
        }

        return metrics_by_name
