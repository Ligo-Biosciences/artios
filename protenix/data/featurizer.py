import copy
from collections import defaultdict
from typing import Union

import numpy as np
import torch
from biotite.structure import AtomArray, get_residue_starts
from sklearn.neighbors import KDTree  # TODO: switch to scipy

from protenix.common.constants import (
    BACKBONE_DEFS,
    STD_RESIDUES,
    STD_RESIDUES_AND_GAP,
    get_all_elems,
)
from protenix.data.features import (
    AtomPermInfo,
    BasicContext,
    ChainPermContext,
    CoordinateContext,
    EntityContext,
    FullContext,
    IndexContext,
    MaskContext,
    MotifContext,
    ReferenceConformerContext,
    TokenContext,
)
from protenix.data.tokenizer import TokenArray
from protenix.utils.geometry import random_transform


class ContextGenerator(object):
    def __init__(
        self,
        cropped_token_array: TokenArray,
        cropped_atom_array: AtomArray,
        pdb_id: str,
        bioassembly_dict_fpath: str,
        ref_pos_augment: bool = True,
        lig_atom_rename: bool = False,
    ) -> None:
        """
        Args:
            cropped_token_array (TokenArray): TokenArray object after cropping
            cropped_atom_array (AtomArray): AtomArray object after cropping
            ref_pos_augment (bool): Boolean indicating whether apply random rotation and translation on ref_pos
            lig_atom_rename (bool): Boolean indicating whether rename atom name for ligand atoms
        """
        self.cropped_token_array = cropped_token_array
        self.cropped_atom_array = cropped_atom_array
        self.pdb_id = pdb_id
        self.bioassembly_dict_fpath = bioassembly_dict_fpath
        self.ref_pos_augment = ref_pos_augment
        self.lig_atom_rename = lig_atom_rename

    @property
    def centre_atoms_indices(self) -> np.ndarray:
        return self.cropped_token_array.get_annotation("centre_atom_index")

    @property
    def centre_atoms(self) -> AtomArray:
        return self.cropped_atom_array[self.centre_atoms_indices]

    @property
    def atom_to_token_idx(self) -> torch.Tensor:
        atom_to_token_idx_dict = {}
        for idx, token in enumerate(self.cropped_token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_to_token_idx_dict[atom_idx] = idx

        # Ensure the order of the atom_to_token_idx is the same as the atom_array
        atom_to_token_idx = torch.Tensor(
            [
                atom_to_token_idx_dict[atom_idx]
                for atom_idx in range(len(self.cropped_atom_array))
            ]
        ).long()
        return atom_to_token_idx

    @property
    def backbone_atoms_tensor(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract backbone atom coordinates and masks for all tokens.

        The backbone atoms depend on the chain type:
        - Proteins: N, CA, C, O, and virtual Cß (computed from N, CA, C)
        - Nucleic acids: OP1, OP2, C1', C3', C4'
        - Ligands: Centre atom only (other slots are zero)

        Returns:
            backbone_coords: Backbone atom coordinates [n_token, 5, 3]
            backbone_mask: Mask indicating valid backbone atoms [n_token, 5]
        """
        # Backbone atom definitions by molecule type
        n_tokens = len(self.cropped_token_array)
        backbone_coords = np.zeros((n_tokens, 5, 3), dtype=np.float32)
        backbone_mask = np.zeros((n_tokens, 5), dtype=np.float32)

        # TODO: vectorized op with masking might be more appropriate here
        # we do this in other parts of the codebase.
        # Make this biotite native.

        for token_idx, token in enumerate(self.cropped_token_array):
            centre_atom = self.centre_atoms[token_idx]
            mol_type = centre_atom.mol_type
            is_standard = centre_atom.res_name in STD_RESIDUES

            # Ligands or non-standard residues: use centre atom only
            if mol_type == "ligand" or not is_standard:
                backbone_coords[token_idx, 0] = centre_atom.coord
                backbone_mask[token_idx, 0] = 1.0
                continue

            # Standard protein or nucleic acid
            backbone_names = BACKBONE_DEFS[mol_type]
            atom_name_map = {name: idx for idx, name in enumerate(token.atom_names)}
            atom_coords = {}

            # Extract backbone atoms
            for bb_idx, atom_name in enumerate(backbone_names):
                if atom_name in atom_name_map:
                    local_idx = atom_name_map[atom_name]
                    global_idx = token.atom_indices[local_idx]
                    backbone_coords[token_idx, bb_idx] = self.cropped_atom_array.coord[
                        global_idx
                    ]
                    backbone_mask[token_idx, bb_idx] = 1.0
                    atom_coords[atom_name] = self.cropped_atom_array.coord[global_idx]

            # Compute virtual Cβ for proteins
            if mol_type == "protein" and {"N", "CA", "C"}.issubset(atom_coords):
                cb_coord = ContextGenerator.compute_virtual_cb(
                    atom_coords["N"], atom_coords["CA"], atom_coords["C"]
                )
                backbone_coords[token_idx, 4] = cb_coord
                backbone_mask[token_idx, 4] = 1.0

        return torch.Tensor(backbone_coords), torch.Tensor(backbone_mask)

    @staticmethod
    def encoder(encode_def_list: list[str], input_list: list[str]) -> torch.Tensor:
        """
        Encode a list of input values into a binary format using a specified encoding definition list.

        Args:
            encode_def_list (list): A list of encoding definitions.
            input_list (list): A list of input values to be encoded.

        Returns:
            torch.Tensor: A tensor representing the binary encoding of the input values.
        """
        onehot_dict = {}
        num_keys = len(encode_def_list)
        for index, key in enumerate(encode_def_list):
            onehot = [0] * num_keys
            onehot[index] = 1
            onehot_dict[key] = onehot

        onehot_encoded_data = [onehot_dict[item] for item in input_list]
        onehot_tensor = torch.Tensor(onehot_encoded_data).long()
        return onehot_tensor

    @staticmethod
    def restype_onehot_encoded(restype_list: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "restype"
        One-hot encoding of the sequence. 32 possible values: 20 amino acids + unknown,
        4 RNA nucleotides + unknown, 4 DNA nucleotides + unknown, and gap.
        Ligands represented as "unknown amino acid".

        Args:
            restype_list (List[str]): A list of residue types.
                                      The residue type of ligand should be "UNK" in the input list.

        Returns:
            torch.Tensor:  A Tensor of one-hot encoded residue types
        """

        return ContextGenerator.encoder(list(STD_RESIDUES_AND_GAP.keys()), restype_list)

    @staticmethod
    def elem_onehot_encoded(elem_list: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "ref_element"
        One-hot encoding of the element atomic number for each atom
        in the reference conformer, up to atomic number 128.

        Args:
            elem_list (List[str]): A list of element symbols.

        Returns:
            torch.Tensor:  A Tensor of one-hot encoded elements
        """
        return ContextGenerator.encoder(get_all_elems(), elem_list)

    @staticmethod
    def ref_atom_name_chars_encoded(atom_names: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "ref_atom_name_chars"
        One-hot encoding of the unique atom names in the reference conformer.
        Each character is encoded as ord(c) − 32, and names are padded to length 4.

        Args:
            atom_name_list (List[str]): A list of atom names.

        Returns:
            torch.Tensor:  A Tensor of character encoded atom names
        """
        onehot_dict = {}
        for index, key in enumerate(range(64)):
            onehot = [0] * 64
            onehot[index] = 1
            onehot_dict[key] = onehot
        # [N_atom, 4, 64]
        mol_encode = []
        for atom_name in atom_names:
            # [4, 64]
            atom_encode = []
            for name_str in atom_name.ljust(4):
                atom_encode.append(onehot_dict[ord(name_str) - 32])
            mol_encode.append(atom_encode)
        onehot_tensor = torch.Tensor(mol_encode).long()
        return onehot_tensor

    @staticmethod
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
            Virtual Cβ coordinates [..., 3]

        Reference:
            ProteinMPNN implementation of virtual Cβ placement
        # TODO: not sure this is where this function belongs.
        """
        # Backbone bond vectors
        n_ca = ca_coord - n_coord
        ca_c = c_coord - ca_coord

        # Cross product gives direction perpendicular to peptide plane
        cross = np.cross(n_ca, ca_c)

        # Idealized Cß
        cb_coord = ca_coord + 0.568 * n_ca - 0.54068 * ca_c - 0.5827 * cross

        return cb_coord

    def get_token_context(self) -> TokenContext:
        """
        Ref: AlphaFold3 SI Chapter 2.8

        Get token context.
        The size of these features is [N_token] or [N_token, N_token].

        Returns:
            TokenContext
        """
        restype = self.centre_atoms.cano_seq_resname
        restype_onehot = self.restype_onehot_encoded(restype)

        context = TokenContext(
            token_index=torch.arange(0, len(self.cropped_token_array)),
            residue_index=torch.Tensor(self.centre_atoms.res_id.astype(int)).long(),
            asym_id=torch.Tensor(self.centre_atoms.asym_id_int).long(),
            entity_id=torch.Tensor(self.centre_atoms.entity_id_int).long(),
            sym_id=torch.Tensor(self.centre_atoms.sym_id_int).long(),
            restype=restype_onehot,
            token_bonds=self.get_bonds(),
        )
        return context

    def get_chain_perm_context(self) -> ChainPermContext:
        """
        The chain permutation use "entity_mol_id", "mol_id" and "mol_atom_index"
        instead of the "entity_id", "asym_id" and "residue_index".

        The shape of these features is [N_atom].

        Returns:
            ChainPermContext
        """
        chain_perm_context = ChainPermContext(
            mol_id=torch.Tensor(self.cropped_atom_array.mol_id).long(),
            mol_atom_index=torch.Tensor(self.cropped_atom_array.mol_atom_index).long(),
            entity_mol_id=torch.Tensor(self.cropped_atom_array.entity_mol_id).long(),
        )
        return chain_perm_context

    def get_renamed_atom_names(self) -> np.ndarray:
        """
        Rename the atom names of ligands to avoid information leakage.

        Returns:
            np.ndarray: A numpy array of renamed atom names.
        """
        res_starts = get_residue_starts(
            self.cropped_atom_array, add_exclusive_stop=True
        )
        new_atom_names = copy.deepcopy(self.cropped_atom_array.atom_name)
        for start, stop in zip(res_starts[:-1], res_starts[1:]):
            res_mol_type = self.cropped_atom_array.mol_type[start]
            if res_mol_type != "ligand":
                continue

            elem_count = defaultdict(int)
            new_res_atom_names = []
            for elem in self.cropped_atom_array.element[start:stop]:
                elem_count[elem] += 1
                new_res_atom_names.append(f"{elem.upper()}{elem_count[elem]}")
            new_atom_names[start:stop] = new_res_atom_names
        return new_atom_names

    def get_reference_conformer_context(self) -> ReferenceConformerContext:
        """
        Ref: AlphaFold3 SI Chapter 2.8

        Get reference features.

        Returns:
            ReferenceConformerContext
        """
        ref_pos = []
        for ref_space_uid in np.unique(self.cropped_atom_array.ref_space_uid):
            res_ref_pos = random_transform(
                self.cropped_atom_array.ref_pos[
                    self.cropped_atom_array.ref_space_uid == ref_space_uid,
                ],
                apply_augmentation=self.ref_pos_augment,
                centralize=True,
            )
            ref_pos.append(res_ref_pos)
        ref_pos = np.concatenate(ref_pos)

        if self.lig_atom_rename:
            atom_names = self.get_renamed_atom_names()
        else:
            atom_names = self.cropped_atom_array.atom_name

        ref_conf_context = ReferenceConformerContext(
            ref_pos=torch.Tensor(ref_pos),
            ref_mask=torch.Tensor(self.cropped_atom_array.ref_mask).long(),
            ref_element=ContextGenerator.elem_onehot_encoded(
                self.cropped_atom_array.element
            ).long(),
            ref_charge=torch.Tensor(self.cropped_atom_array.ref_charge).long(),
            ref_atom_name_chars=ContextGenerator.ref_atom_name_chars_encoded(
                atom_names
            ).long(),
            ref_space_uid=torch.Tensor(self.cropped_atom_array.ref_space_uid).long(),
        )

        return ref_conf_context

    def get_bonds(self) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Chapter 2.8
        A 2D matrix indicating if there is a bond between any atom in token i and token j,
        restricted to just polymer-ligand and ligand-ligand bonds and bonds less than 2.4 Å during training.

        Returns:
            torch.Tensor of shape [n_token, n_token] indicating bonds between tokens
        """
        num_tokens = len(self.cropped_token_array)
        adj_matrix = self.cropped_atom_array.bonds.adjacency_matrix().astype(int)

        token_adj_matrix = np.zeros((num_tokens, num_tokens), dtype=int)
        atom_bond_mask = adj_matrix > 0

        for i in range(num_tokens):
            atoms_i = self.cropped_token_array[i].atom_indices
            token_i_mol_type = self.cropped_atom_array.mol_type[atoms_i[0]]
            token_i_res_name = self.cropped_atom_array.res_name[atoms_i[0]]
            token_i_ref_space_uid = self.cropped_atom_array.ref_space_uid[atoms_i[0]]
            unstd_res_token_i = (
                token_i_res_name not in STD_RESIDUES and token_i_mol_type != "ligand"
            )
            is_polymer_i = token_i_mol_type in ["protein", "dna", "rna"]

            for j in range(i + 1, num_tokens):
                atoms_j = self.cropped_token_array[j].atom_indices
                token_j_mol_type = self.cropped_atom_array.mol_type[atoms_j[0]]
                token_j_res_name = self.cropped_atom_array.res_name[atoms_j[0]]
                token_j_ref_space_uid = self.cropped_atom_array.ref_space_uid[
                    atoms_j[0]
                ]
                unstd_res_token_j = (
                    token_j_res_name not in STD_RESIDUES
                    and token_j_mol_type != "ligand"
                )
                is_polymer_j = token_j_mol_type in ["protein", "dna", "rna"]

                # The polymer-polymer (std-std, std-unstd, and inter-unstd) bond will not be included in token_bonds.
                if is_polymer_i and is_polymer_j:
                    is_same_res = token_i_ref_space_uid == token_j_ref_space_uid
                    unstd_res_bonds = unstd_res_token_i and unstd_res_token_j
                    if not (is_same_res and unstd_res_bonds):
                        continue

                sub_matrix = atom_bond_mask[np.ix_(atoms_i, atoms_j)]
                if np.any(sub_matrix):
                    token_adj_matrix[i, j] = 1
                    token_adj_matrix[j, i] = 1
        return torch.Tensor(token_adj_matrix).long()

    def get_index_context(self) -> IndexContext:
        """Get atom-to-token index mappings and global features.

        Returns:
            IndexContext: Atom-to-token mappings.
        """
        return IndexContext(
            atom_to_token_idx=self.atom_to_token_idx,
            atom_to_tokatom_idx=torch.Tensor(
                self.cropped_atom_array.tokatom_idx
            ).long(),
        )

    def get_entity_context(self) -> EntityContext:
        """Get molecule type indicators at atom and token level.

        Returns:
            EntityContext: Boolean masks for molecule types.
        """

        def get_token_level_moltype(mol_type: str):
            is_x_atom = torch.Tensor(
                getattr(self.cropped_atom_array, f"is_{mol_type}")
            ).to(dtype=torch.bool)
            is_x_token = torch.zeros(len(self.cropped_token_array), dtype=torch.bool)
            is_x_token.scatter_(dim=-1, index=self.atom_to_token_idx, src=is_x_atom)

            return is_x_token

        return EntityContext(
            is_protein=torch.Tensor(self.cropped_atom_array.is_protein).bool(),
            is_ligand=torch.Tensor(self.cropped_atom_array.is_ligand).bool(),
            is_dna=torch.Tensor(self.cropped_atom_array.is_dna).bool(),
            is_rna=torch.Tensor(self.cropped_atom_array.is_rna).bool(),
            is_protein_token=get_token_level_moltype("protein"),
            is_ligand_token=get_token_level_moltype("ligand"),
            is_dna_token=get_token_level_moltype("dna"),
            is_rna_token=get_token_level_moltype("rna"),
        )

    def get_mask_context(self) -> MaskContext:
        """Generate mask features for the cropped atom array.

        Returns:
            MaskContext: Atom-level masks for loss computation and attention.
        """
        atom_is_resolved = self.cropped_atom_array.is_resolved.astype(int)
        atom_in_token_w_resolved_centre_atom = self.centre_atoms.is_resolved[
            self.atom_to_token_idx
        ].astype(int)

        return MaskContext(
            centre_atom_mask=torch.Tensor(
                self.cropped_atom_array.centre_atom_mask
            ).bool(),
            atom_mask=torch.ones(len(self.cropped_atom_array)).bool(),
            is_resolved=torch.Tensor(self.cropped_atom_array.is_resolved).bool(),
            backbone_mask=torch.Tensor(self.backbone_atoms_tensor[1]).bool(),
            atom_loss_mask=torch.Tensor(
                atom_is_resolved * atom_in_token_w_resolved_centre_atom
            ).bool(),
            token_mask=torch.ones(len(self.cropped_token_array)).bool(),
        )

    def get_motif_context(self) -> MotifContext:
        """Get features for motif training.

        We extract coordinates from the centre atoms to avoid creating heavy
        N_atom features when this method is called for all-atom models.

        Returns:
            MotifContext: Motif features for motif scaffolding.
        """
        is_motif = self.cropped_atom_array.is_motif
        is_motif_related = self.cropped_atom_array.is_motif_related

        token_resolved = self.centre_atoms.is_resolved  # [N_token]
        is_motif_token = self.centre_atoms.is_motif  # [N_token]
        is_motif_related_token = self.centre_atoms.is_motif_related  # [N_token]

        restype = self.centre_atoms.cano_seq_resname
        restype_onehot = self.restype_onehot_encoded(restype)

        # TODO: this part is a bit ugly with the reshapes etc.
        motif_restype = restype_onehot * torch.Tensor(
            is_motif_related_token * token_resolved
        ).unsqueeze(-1)  # [N_token, 32]

        motif_coordinate = self.centre_atoms.coord * is_motif_token.reshape(
            -1, 1
        )  # [N_token, 3]

        return MotifContext(
            is_motif=torch.Tensor(is_motif).bool(),
            is_motif_related=torch.Tensor(is_motif_related).bool(),
            is_motif_token=torch.Tensor(is_motif_token).long(),
            is_motif_related_token=torch.Tensor(is_motif_related_token).long(),
            motif_restype=motif_restype.float(),
            motif_coordinate=torch.Tensor(motif_coordinate).float(),
        )

    def get_coordinate_context(self) -> CoordinateContext:
        """Get coordinates features.

        Returns:
            CoordinateContext: Coordinates features.
        """
        return CoordinateContext(
            coordinate=torch.Tensor(self.cropped_atom_array.coord).float(),
            centre_coordinate=torch.Tensor(self.centre_atoms.coord),
            backbone_coordinate=self.backbone_atoms_tensor[0],
            resolution=float(self.cropped_atom_array.resolution[0]),
        )

    def get_basic_context(self) -> BasicContext:
        """Get basic context information."""
        return BasicContext(
            atom_array=self.cropped_atom_array,
            token_array=self.cropped_token_array,
            pdb_id=self.pdb_id,
            bioassembly_dict_fpath=self.bioassembly_dict_fpath,
            n_asym=len(np.unique(self.centre_atoms.asym_id_int)),
            n_token=len(self.cropped_token_array),
            n_atom=len(self.cropped_atom_array),
        )

    def get_full_context(self) -> FullContext:
        """Get all context required for design pipeline."""
        return FullContext(
            token=self.get_token_context(),
            reference=self.get_reference_conformer_context(),
            chain_perm=self.get_chain_perm_context(),
            mask=self.get_mask_context(),
            motif=self.get_motif_context(),
            index=self.get_index_context(),
            entity=self.get_entity_context(),
            coordinate=self.get_coordinate_context(),
            basic=self.get_basic_context(),
        )

    def get_atom_permutation_list(
        self,
    ) -> AtomPermInfo:
        """
        Generate info of permutations.

        Returns:
            List[List[int]]: a list of atom permutations.

        # TODO: add this back in.
        """
        atom_perm_list = []
        for i in self.cropped_atom_array.res_perm:
            # Decode list[str] -> list[list[int]]
            atom_perm_list.append([int(j) for j in i.split("_")])

        # Atoms connected to different residue are fixed.
        # Bonds array: [[atom_idx_i, atom_idx_j, bond_type]]
        idx_i = self.cropped_atom_array.bonds._bonds[:, 0]
        idx_j = self.cropped_atom_array.bonds._bonds[:, 1]
        diff_mask = (
            self.cropped_atom_array.ref_space_uid[idx_i]
            != self.cropped_atom_array.ref_space_uid[idx_j]
        )
        inter_residue_bonds = self.cropped_atom_array.bonds._bonds[diff_mask]
        fixed_atom_mask = np.isin(
            np.arange(len(self.cropped_atom_array)),
            np.unique(inter_residue_bonds[:, :2]),
        )

        # Get fixed atom permutation for each residue.
        fixed_atom_perm_list = []
        res_starts = get_residue_starts(
            self.cropped_atom_array, add_exclusive_stop=True
        )
        for r_start, r_stop in zip(res_starts[:-1], res_starts[1:]):
            atom_res_perm = np.array(
                atom_perm_list[r_start:r_stop]
            )  # [N_res_atoms, N_res_perm]
            res_fixed_atom_mask = fixed_atom_mask[r_start:r_stop]

            if np.sum(res_fixed_atom_mask) == 0:
                # If all atoms in the residue are not fixed, e.g. ions
                fixed_atom_perm_list.extend(atom_res_perm.tolist())
                continue

            # Create a [N_res_atoms, N_res_perm] template of indices
            n_res_atoms, n_perm = atom_res_perm.shape
            indices_template = (
                atom_res_perm[:, 0].reshape(n_res_atoms, 1).repeat(n_perm, axis=1)
            )

            # Identify the column where the positions of the fixed atoms remain unchanged
            fixed_atom_perm = atom_res_perm[
                res_fixed_atom_mask
            ]  # [N_fixed_res_atoms, N_res_perm]
            fixed_indices_template = indices_template[
                res_fixed_atom_mask
            ]  # [N_fixed_res_atoms, N_res_perm]
            unchanged_columns_mask = np.all(
                fixed_atom_perm == fixed_indices_template, axis=0
            )

            # Remove the columns related to the position changes of fixed atoms.
            fiedx_atom_res_perm = atom_res_perm[:, unchanged_columns_mask]
            fixed_atom_perm_list.extend(fiedx_atom_res_perm.tolist())
        return AtomPermInfo(perms=fixed_atom_perm_list)

    @staticmethod
    def get_lig_pocket_mask(
        atom_array: AtomArray, lig_label_asym_id: Union[str, list]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ref: AlphaFold3 Chapter Methods.Metrics

        the pocket is defined as all heavy atoms within 10 Å of any heavy atom of the ligand,
        restricted to the primary polymer chain for the ligand or modified residue being scored,
        and further restricted to only backbone atoms for proteins. The primary polymer chain is defined variously:
        for PoseBusters it is the protein chain with the most atoms within 10 Å of the ligand,
        for bonded ligand scores it is the bonded polymer chain and for modified residues it
        is the chain that the residue is contained in (minus that residue).

        Args:
            atom_array (AtomArray): atoms in the complex.
            lig_label_asym_id (Union[str, List]): The label_asym_id of the ligand of interest.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of ligand pocket mask and pocket mask.

        # TODO: this is useful, but we need to adapt it to our features.
        """

        if isinstance(lig_label_asym_id, str):
            lig_label_asym_ids = [lig_label_asym_id]
        else:
            lig_label_asym_ids = list(lig_label_asym_id)

        # Get backbone mask
        prot_backbone = (
            atom_array.is_protein & np.isin(atom_array.atom_name, ["C", "N", "CA"])
        ).astype(bool)

        kdtree = KDTree(atom_array.coord)

        ligand_mask_list = []
        pocket_mask_list = []
        for lig_label_asym_id in lig_label_asym_ids:
            assert np.isin(
                lig_label_asym_id, atom_array.label_asym_id
            ), f"{lig_label_asym_id} is not in the label_asym_id of the cropped atom array."

            ligand_mask = atom_array.label_asym_id == lig_label_asym_id
            lig_pos = atom_array.coord[ligand_mask]

            # Get atoms in 10 Angstrom radius
            near_atom_indices = np.unique(
                np.concatenate(kdtree.query_radius(lig_pos, 10.0))
            )
            near_atoms = [
                True if i in near_atom_indices else False
                for i in range(len(atom_array))
            ]

            # Get primary chain (protein backone in 10 Angstrom radius)
            primary_chain_candidates = near_atoms & prot_backbone
            primary_chain_candidates_atoms = atom_array[primary_chain_candidates]

            max_atom = 0
            primary_chain_asym_id_int = None
            for asym_id_int in np.unique(primary_chain_candidates_atoms.asym_id_int):
                n_atoms = np.sum(
                    primary_chain_candidates_atoms.asym_id_int == asym_id_int
                )
                if n_atoms > max_atom:
                    max_atom = n_atoms
                    primary_chain_asym_id_int = asym_id_int
            assert (
                primary_chain_asym_id_int is not None
            ), f"No primary chain found for ligand ({lig_label_asym_id=})."

            pocket_mask = primary_chain_candidates & (
                atom_array.asym_id_int == primary_chain_asym_id_int
            )
            ligand_mask_list.append(ligand_mask)
            pocket_mask_list.append(pocket_mask)

        ligand_mask_by_pockets = torch.Tensor(
            np.array(ligand_mask_list).astype(int)
        ).long()
        pocket_mask_by_pockets = torch.Tensor(
            np.array(pocket_mask_list).astype(int)
        ).long()
        return ligand_mask_by_pockets, pocket_mask_by_pockets
