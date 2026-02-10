"""
Houses all the logic for generating fragments from protein structures.
"""

from dataclasses import dataclass
from logging import getLogger

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial import KDTree

logger = getLogger(__name__)


@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    avg_plddt: float
    length: int
    rg: float
    sasa: float
    boundaries: str


class FragmentMaker:
    """Extracts compact, well-folded fragments from full-length predicted structures.

    Pipeline: filter_low_plddt_residues -> spectral_split -> assign_fragment_ids
    -> spatially_cluster_fragments -> filter_fragments

    See docs/fragment_detection.md for algorithm details.
    """

    def __init__(
        self,
        plddt_disorder_threshold: float = 65,
        contact_distance_threshold: float = 4.0,
        min_contact_residues: int = 4,
        min_fragment_length: int = 150,
        min_fragment_plddt: float = 70,
        spectral_k: int = 15,
        spectral_sigma: float = 8.0,
        spectral_connectivity_threshold: float = 0.05,
        spectral_boundary_size: int = 4,
        spectral_min_partition_size: int = 50,
    ):
        """
        Initialize FragmentMaker with filtering and clustering thresholds.

        Args:
            plddt_disorder_threshold (float): Threshold for pLDDT to filter residues. Defaults to 65.
            contact_distance_threshold (float): Distance threshold for atom-atom contacts in spatial clustering. Defaults to 4.0 Angstroms.
            min_contact_residues (int): Minimum number of residues with contacts to merge fragments. Defaults to 4.
            min_fragment_length (int): Minimum length of a fragment. Defaults to 150.
            min_fragment_plddt (float): Minimum pLDDT of a fragment. Defaults to 70.
            spectral_k (int): Number of nearest neighbors for kNN graph in spectral bisection. Defaults to 15.
            spectral_sigma (float): Gaussian kernel bandwidth for edge weights. Defaults to 8.0.
            spectral_connectivity_threshold (float): Fiedler value (lambda_2) above which a partition is considered well-connected. Defaults to 0.05.
            spectral_boundary_size (int): Number of residues to remove at each bisection boundary. Defaults to 4.
            spectral_min_partition_size (int): Minimum partition size to attempt further bisection. Defaults to 50.
        """
        self.plddt_residue_threshold = plddt_disorder_threshold
        self.contact_distance_threshold = contact_distance_threshold
        self.min_contact_residues = min_contact_residues
        self.min_fragment_length = min_fragment_length
        self.min_fragment_plddt = min_fragment_plddt
        self.spectral_k = spectral_k
        self.spectral_sigma = spectral_sigma
        self.spectral_connectivity_threshold = spectral_connectivity_threshold
        self.spectral_boundary_size = spectral_boundary_size
        self.spectral_min_partition_size = spectral_min_partition_size

        # Create length-dependent Rg upper bound
        self.rg_upper_bound_line = (
            0.029767441860465118,
            14.279069767441861,
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

    def spectral_split(self, atom_array: AtomArray) -> AtomArray:
        """
        Recursive spectral bisection to detect linker residues between domains.

        Builds a Gaussian-weighted kNN graph over CA atoms (excluding those
        already filtered by pLDDT), computes the normalized Laplacian, and
        recursively bisects using the Fiedler vector. Residues at the sign
        change boundary of each bisection are marked as linkers (fragment_id = -1).

        Args:
            atom_array (AtomArray): AtomArray with fragment_id annotation initialized
                (fragment_id = -1 for filtered residues, 0 for valid ones).

        Returns:
            AtomArray: AtomArray with linker residues additionally marked as -1.
        """
        ca_mask = atom_array.atom_name == "CA"
        valid_mask = ca_mask & (atom_array.fragment_id != -1)
        ca_indices = np.where(valid_mask)[0]

        if len(ca_indices) < self.spectral_min_partition_size:
            return atom_array

        coords = atom_array.coord[ca_indices]
        n = len(coords)

        # Build Gaussian-weighted kNN adjacency matrix (sparse + vectorized)
        tree = KDTree(coords)
        k = min(self.spectral_k, n - 1)
        distances, neighbors = tree.query(coords, k=k + 1)

        # Skip self-loops (column 0), vectorize weight computation
        row = np.repeat(np.arange(n), k)
        col = neighbors[:, 1:].ravel()
        weights = np.exp(
            -(distances[:, 1:] ** 2).ravel() / (2 * self.spectral_sigma**2)
        )
        W = csr_matrix((weights, (row, col)), shape=(n, n))
        W = W.maximum(W.T)  # symmetrize with element-wise max

        linker_local_indices: set[int] = set()

        def _split(node_indices: list[int], depth: int = 0) -> None:
            if len(node_indices) < self.spectral_min_partition_size:
                return

            idx = np.array(sorted(node_indices))
            sub_W = W[idx][:, idx]

            # Compute D^{-1/2} W D^{-1/2} (= I - L_sym) as sparse
            d = np.asarray(sub_W.sum(axis=1)).ravel()
            d_inv_sqrt = np.zeros_like(d)
            nonzero = d > 0
            d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])

            L_norm = diags(d_inv_sqrt) @ sub_W @ diags(d_inv_sqrt)

            # Find 2 largest eigenvalues of L_norm; fiedler_value = 1 - λ₂(L_norm)
            evals, evecs = eigsh(L_norm, k=2, which="LM")
            fiedler_value = 1.0 - evals[0]
            fiedler_vector = evecs[:, 0]

            if fiedler_value > self.spectral_connectivity_threshold:
                return

            # Remove boundary_size residues closest to sign change
            fiedler_abs = np.abs(fiedler_vector)
            boundary_order = np.argsort(fiedler_abs)[: self.spectral_boundary_size]
            boundary_set = set(boundary_order.tolist())

            for bi in boundary_order:
                linker_local_indices.add(idx[bi])

            # Partition remaining nodes by Fiedler sign
            partition_a = [
                idx[i]
                for i in range(len(idx))
                if i not in boundary_set and fiedler_vector[i] < 0
            ]
            partition_b = [
                idx[i]
                for i in range(len(idx))
                if i not in boundary_set and fiedler_vector[i] >= 0
            ]

            _split(partition_a, depth + 1)
            _split(partition_b, depth + 1)

        _split(list(range(n)))

        if linker_local_indices:
            fragment_ids = atom_array.fragment_id.copy()
            linker_atom_indices = ca_indices[np.array(sorted(linker_local_indices))]
            linker_res_ids = set(atom_array.res_id[linker_atom_indices].tolist())
            mask = np.isin(atom_array.res_id, list(linker_res_ids))
            fragment_ids[mask] = -1
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

    def fill_spectral_gaps(self, atom_array: AtomArray) -> AtomArray:
        """
        Reclaim boundary residues that create internal gaps within fragments.

        After spatial clustering re-merges fragments that were split by spectral
        bisection, the removed boundary residues leave small internal gaps. This
        method fills those gaps by reclaiming residues that:
        1. Sit inside a gap of at most 2 * spectral_boundary_size residues
        2. Have pLDDT above the disorder threshold (confirming they were removed
           by spectral_split, not by pLDDT filtering)

        Args:
            atom_array (AtomArray): AtomArray with fragment_id annotation after
                spatial clustering.

        Returns:
            AtomArray: AtomArray with small internal gaps filled.
        """
        fragment_ids = atom_array.fragment_id.copy()
        max_gap_size = 2 * self.spectral_boundary_size

        unique_frag_ids = np.unique(fragment_ids)
        unique_frag_ids = unique_frag_ids[unique_frag_ids != -1]

        for frag_id in unique_frag_ids:
            frag_res_ids = np.unique(atom_array.res_id[fragment_ids == frag_id])
            sorted_res = np.sort(frag_res_ids)

            # Find internal gaps
            diffs = np.diff(sorted_res)
            gap_indices = np.where(diffs > 1)[0]

            for gi in gap_indices:
                gap_start = sorted_res[gi] + 1
                gap_end = sorted_res[gi + 1]
                gap_size = gap_end - gap_start

                if gap_size > max_gap_size:
                    continue

                # Check that all gap residues have good pLDDT
                all_reclaimable = True
                for res_id in range(gap_start, gap_end):
                    res_mask = atom_array.res_id == res_id
                    if not np.any(res_mask):
                        all_reclaimable = False
                        break
                    if (
                        np.mean(atom_array.b_factor[res_mask])
                        < self.plddt_residue_threshold
                    ):
                        all_reclaimable = False
                        break

                if all_reclaimable:
                    for res_id in range(gap_start, gap_end):
                        res_mask = atom_array.res_id == res_id
                        fragment_ids[res_mask] = frag_id

        atom_array.set_annotation("fragment_id", fragment_ids)
        return atom_array

    def _calculate_fragment_metrics_fast(self, atom_array: AtomArray) -> list[Fragment]:
        """
        Calculate cheap per-fragment metrics (pLDDT, length, Rg, boundaries).

        Skips SASA computation (expensive). Used internally by filter_fragments
        where SASA is not needed for filtering decisions.

        Args:
            atom_array (AtomArray): AtomArray with fragment_id annotation.

        Returns:
            list[Fragment]: List of Fragment objects with sasa=0.0, sorted by fragment_id.
        """
        fragments: list[Fragment] = []

        unique_frag_ids = np.unique(atom_array.fragment_id)
        unique_frag_ids = unique_frag_ids[unique_frag_ids != -1]

        for frag_id in unique_frag_ids:
            mask = atom_array.fragment_id == frag_id
            frag_atoms = atom_array[mask]

            avg_plddt = np.mean(frag_atoms.b_factor)

            unique_res_ids = np.unique(frag_atoms.res_id)
            length = len(unique_res_ids)

            centroid = np.mean(frag_atoms.coord, axis=0)
            squared_distances = np.sum((frag_atoms.coord - centroid) ** 2, axis=1)
            rg = np.sqrt(np.mean(squared_distances))

            sorted_res_ids = np.sort(unique_res_ids)
            segments = []
            segment_start = sorted_res_ids[0]
            prev_res_id = sorted_res_ids[0]

            for res_id in sorted_res_ids[1:]:
                if res_id - prev_res_id != 1:
                    segments.append(f"{segment_start}-{prev_res_id}")
                    segment_start = res_id
                prev_res_id = res_id

            segments.append(f"{segment_start}-{prev_res_id}")
            boundaries = "_".join(segments)

            fragments.append(
                Fragment(
                    fragment_id=int(frag_id),
                    avg_plddt=float(avg_plddt),
                    length=length,
                    rg=float(rg),
                    sasa=0.0,
                    boundaries=boundaries,
                )
            )

        return sorted(fragments, key=lambda f: f.fragment_id)

    def calculate_fragment_metrics(self, atom_array: AtomArray) -> list[Fragment]:
        """
        Calculate all per-fragment metrics including SASA.

        Args:
            atom_array (AtomArray): AtomArray with fragment_id annotation.

        Returns:
            list[Fragment]: List of Fragment objects sorted by fragment_id.
        """
        fragments = self._calculate_fragment_metrics_fast(atom_array)

        # Add SASA (expensive) to each fragment
        full_fragments = []
        for frag in fragments:
            mask = atom_array.fragment_id == frag.fragment_id
            frag_atoms = atom_array[mask]
            atom_sasa = struc.sasa(frag_atoms, vdw_radii="Single")
            sasa = float(np.nanmean(atom_sasa))

            full_fragments.append(
                Fragment(
                    fragment_id=frag.fragment_id,
                    avg_plddt=frag.avg_plddt,
                    length=frag.length,
                    rg=frag.rg,
                    sasa=sasa,
                    boundaries=frag.boundaries,
                )
            )

        return full_fragments

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
        fragments = self._calculate_fragment_metrics_fast(atom_array)

        for frag in fragments:
            passes_length = frag.length >= self.min_fragment_length
            passes_plddt = frag.avg_plddt >= self.min_fragment_plddt

            # Calculate length-dependent Rg upper bound using linear equation
            slope, y_intercept = self.rg_upper_bound_line
            max_rg = slope * frag.length + y_intercept
            passes_rg = frag.rg <= max_rg

            if not (passes_length and passes_plddt and passes_rg):
                mask = atom_array.fragment_id == frag.fragment_id
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
        atom_array = self.spectral_split(atom_array)
        atom_array = self.assign_fragment_ids(atom_array)
        atom_array = self.spatially_cluster_fragments(atom_array)
        atom_array = self.fill_spectral_gaps(atom_array)
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
        atom_array = self.spectral_split(atom_array)
        atom_array = self.assign_fragment_ids(atom_array)
        atom_array = self.spatially_cluster_fragments(atom_array)
        atom_array = self.fill_spectral_gaps(atom_array)

        fragments = self.calculate_fragment_metrics(atom_array)

        metrics_by_name = {
            "fragment_avg_plddt": [f.avg_plddt for f in fragments],
            "fragment_length": [f.length for f in fragments],
            "fragment_rg": [f.rg for f in fragments],
            "fragment_sasa": [f.sasa for f in fragments],
            "fragment_boundaries": [f.boundaries for f in fragments],
        }

        return metrics_by_name
