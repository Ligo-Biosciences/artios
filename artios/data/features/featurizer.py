from collections import defaultdict
from functools import cached_property
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from biotite.structure import AtomArray, get_residue_starts
from sklearn.neighbors import KDTree  # TODO: switch to scipy

from artios.common.constants import (
    BACKBONE_DEFS,
    STD_RESIDUES,
    STD_RESIDUES_AND_GAP,
    get_all_elems,
)
from artios.data.features import (
    AtomPermContext,
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
from artios.data.structure import AnnotatedStructure
from artios.utils.geometry import compute_virtual_cb, random_transform


class ContextGenerator:
    def __init__(
        self,
        structure: AnnotatedStructure,
        *,
        pdb_id: str = "unknown",
        bioassembly_dict_fpath: str = "",
        original_motif_mapping: list[dict] | None = None,
        ref_pos_augment: bool = True,
        lig_atom_rename: bool = False,
    ) -> None:
        self.s = structure
        self.pdb_id = pdb_id
        self.bioassembly_dict_fpath = bioassembly_dict_fpath
        self.ref_pos_augment = ref_pos_augment
        self.lig_atom_rename = lig_atom_rename
        self.original_motif_mapping = original_motif_mapping

        self.atom_to_token_idx = torch.from_numpy(
            self.s.grouping.atom_token_index
        ).long()

    @cached_property
    def centre_atom_indices(self) -> np.ndarray:
        return self.s.centre_atom_indices

    @cached_property
    def centre_atoms(self) -> AtomArray:
        return self.s.atoms[self.centre_atom_indices]

    @cached_property
    def backbone_atoms_and_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract backbone atom coordinates and masks for all tokens."""
        return _extract_backbone_atoms(self.s)

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
        num_keys = len(encode_def_list)
        idx_map = {key: idx for idx, key in enumerate(encode_def_list)}
        indices = torch.tensor([idx_map[item] for item in input_list], dtype=torch.long)
        return F.one_hot(indices, num_classes=num_keys).to(torch.long)

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
        # Vectorized: encode all atom names at once via ASCII codes
        n = len(atom_names)
        padded = "".join(name.ljust(4)[:4] for name in atom_names)
        char_codes = np.frombuffer(padded.encode("ascii"), dtype=np.uint8)
        indices = np.clip(char_codes.astype(np.int64) - 32, 0, 63)
        indices_tensor = torch.from_numpy(indices).reshape(n, 4)
        return F.one_hot(indices_tensor, num_classes=64).to(torch.long)

    def get_token_context(self) -> TokenContext:
        """
        Ref: AlphaFold3 SI Chapter 2.8

        Get token context.
        The size of these features is [N_token] or [N_token, N_token].

        Returns:
            TokenContext
        """
        centre = self.centre_atom_indices
        restype = self.s.derived.cano_seq_resname[centre]
        restype_onehot = self.restype_onehot_encoded(restype)

        return TokenContext(
            token_index=torch.arange(self.s.grouping.n_tokens),
            residue_index=torch.from_numpy(
                self.s.atoms.res_id[centre].astype(np.int64)
            ),
            asym_id=torch.from_numpy(self.s.entity_ids.asym_id_int[centre]),
            entity_id=torch.from_numpy(self.s.entity_ids.entity_id_int[centre]),
            sym_id=torch.from_numpy(self.s.entity_ids.sym_id_int[centre]),
            restype=restype_onehot,
            # NOTE: token_bonds disabled - expensive O(n_token^2) computation, currently unused
            # token_bonds=self.get_bonds(),
        )

    def get_chain_perm_context(self) -> ChainPermContext:
        """
        The chain permutation use "entity_mol_id", "mol_id" and "mol_atom_index"
        instead of the "entity_id", "asym_id" and "residue_index".

        The shape of these features is [N_atom].

        Returns:
            ChainPermContext
        """
        mol = self.s.molecule_ids
        return ChainPermContext(
            mol_id=torch.from_numpy(mol.mol_id.astype(np.int64)),
            mol_atom_index=torch.from_numpy(mol.mol_atom_index.astype(np.int64)),
            entity_mol_id=torch.from_numpy(mol.entity_mol_id.astype(np.int64)),
        )

    def get_renamed_atom_names(self) -> np.ndarray:
        """
        Rename the atom names of ligands to avoid information leakage.

        Returns:
            np.ndarray: A numpy array of renamed atom names.
        """
        res_starts = get_residue_starts(self.s.atoms, add_exclusive_stop=True)
        new_atom_names = self.s.atoms.atom_name.copy()
        for start, stop in zip(res_starts[:-1], res_starts[1:]):
            if self.s.type_masks.mol_type[start] != "ligand":
                continue

            elem_count = defaultdict(int)
            new_res_atom_names = []
            for elem in self.s.atoms.element[start:stop]:
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
        ref = self.s.reference
        ref_pos = np.zeros_like(self.s.atoms.coord)
        for uid in np.unique(ref.ref_space_uid):
            mask = ref.ref_space_uid == uid
            ref_pos[mask] = random_transform(
                ref.ref_pos[mask],
                apply_augmentation=self.ref_pos_augment,
                centralize=True,
            )

        if self.lig_atom_rename:
            atom_names = self.get_renamed_atom_names()
        else:
            atom_names = self.s.atoms.atom_name

        return ReferenceConformerContext(
            ref_pos=torch.from_numpy(ref_pos.astype(np.float32)),
            ref_mask=torch.from_numpy(ref.ref_mask.astype(np.int64)),
            ref_element=ContextGenerator.elem_onehot_encoded(
                self.s.atoms.element
            ).long(),
            ref_charge=torch.from_numpy(ref.ref_charge.astype(np.int64)),
            ref_atom_name_chars=ContextGenerator.ref_atom_name_chars_encoded(
                atom_names.tolist()
            ),
            ref_space_uid=torch.from_numpy(ref.ref_space_uid.astype(np.int64)),
        )

    def get_bonds(self) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Chapter 2.8
        A 2D matrix indicating if there is a bond between any atom in token i and token j,
        restricted to just polymer-ligand and ligand-ligand bonds and bonds less than 2.4 Å during training.

        Returns:
            torch.Tensor of shape [n_token, n_token] indicating bonds between tokens
        """
        n_tokens = self.s.grouping.n_tokens
        token_adj_matrix = np.zeros((n_tokens, n_tokens), dtype=np.int64)

        bond_array = self.s.atoms.bonds.as_array()
        if len(bond_array) == 0:
            return torch.from_numpy(token_adj_matrix)

        bond_atom_i = bond_array[:, 0]
        bond_atom_j = bond_array[:, 1]

        mol_type = self.s.type_masks.mol_type
        res_name = self.s.atoms.res_name
        ref_space_uid = self.s.reference.ref_space_uid

        polymer_mask = np.isin(mol_type, ["protein", "dna", "rna"])
        unstd_polymer_mask = polymer_mask & ~np.isin(
            res_name, list(STD_RESIDUES.keys())
        )

        # Exclude polymer-polymer bonds UNLESS both endpoints are non-standard and same ref_space_uid
        both_polymer = polymer_mask[bond_atom_i] & polymer_mask[bond_atom_j]
        both_unstd = unstd_polymer_mask[bond_atom_i] & unstd_polymer_mask[bond_atom_j]
        same_ref_space = ref_space_uid[bond_atom_i] == ref_space_uid[bond_atom_j]
        exclude_mask = both_polymer & ~(both_unstd & same_ref_space)

        kept_i = bond_atom_i[~exclude_mask]
        kept_j = bond_atom_j[~exclude_mask]

        # Map atom indices to token indices
        atom_to_token = self.s.grouping.atom_token_index
        token_i = atom_to_token[kept_i]
        token_j = atom_to_token[kept_j]

        # Filter out intra-token bonds
        inter_token = token_i != token_j
        token_i = token_i[inter_token]
        token_j = token_j[inter_token]

        token_adj_matrix[token_i, token_j] = 1
        token_adj_matrix[token_j, token_i] = 1
        return torch.from_numpy(token_adj_matrix)

    def get_index_context(self) -> IndexContext:
        """Get atom-to-token index mappings and global features.

        Returns:
            IndexContext: Atom-to-token mappings.
        """
        return IndexContext(
            atom_to_token_idx=self.atom_to_token_idx,
            atom_to_tokatom_idx=torch.from_numpy(
                self.s.derived.tokatom_idx.astype(np.int64)
            ),
            atom_asym_id=torch.from_numpy(
                self.s.entity_ids.asym_id_int.astype(np.int64)
            ),
        )

    def get_entity_context(self) -> EntityContext:
        """Get molecule type indicators at atom and token level.

        Returns:
            EntityContext: Boolean masks for molecule types.
        """
        tm = self.s.type_masks
        n_tokens = self.s.grouping.n_tokens

        def scatter_to_tokens(atom_mask: np.ndarray) -> torch.Tensor:
            src = torch.from_numpy(atom_mask.astype(bool))
            out = torch.zeros(n_tokens, dtype=torch.bool)
            out.scatter_(dim=-1, index=self.atom_to_token_idx, src=src)
            return out

        return EntityContext(
            is_protein=torch.from_numpy(tm.is_protein.astype(bool)),
            is_ligand=torch.from_numpy(tm.is_ligand.astype(bool)),
            is_dna=torch.from_numpy(tm.is_dna.astype(bool)),
            is_rna=torch.from_numpy(tm.is_rna.astype(bool)),
            is_protein_token=scatter_to_tokens(tm.is_protein),
            is_ligand_token=scatter_to_tokens(tm.is_ligand),
            is_dna_token=scatter_to_tokens(tm.is_dna),
            is_rna_token=scatter_to_tokens(tm.is_rna),
        )

    def get_mask_context(self) -> MaskContext:
        """Generate mask features for the cropped atom array.

        Returns:
            MaskContext: Atom-level masks for loss computation and attention.
        """
        atom_is_resolved = self.s.training.is_resolved.astype(int)
        centre_is_resolved = self.s.training.is_resolved[
            self.centre_atom_indices
        ].astype(int)
        atom_in_resolved_token = centre_is_resolved[self.atom_to_token_idx]

        return MaskContext(
            centre_atom_mask=torch.from_numpy(self.s.grouping.centre_atom_mask),
            atom_mask=torch.ones(len(self.s.atoms), dtype=torch.bool),
            is_resolved=torch.from_numpy(self.s.training.is_resolved.astype(bool)),
            centre_atom_is_resolved=torch.from_numpy(centre_is_resolved.astype(bool)),
            backbone_mask=torch.Tensor(self.backbone_atoms_and_mask[1]).bool(),
            atom_loss_mask=torch.from_numpy(
                (atom_is_resolved * atom_in_resolved_token).astype(bool)
            ),
            token_mask=torch.ones(self.s.grouping.n_tokens, dtype=torch.bool),
        )

    def get_motif_context(self) -> MotifContext:
        """Get features for motif training.

        We extract coordinates from the centre atoms to avoid creating heavy
        N_atom features when this method is called for all-atom models.

        Returns:
            MotifContext: Motif features for motif scaffolding.
        """
        centre = self.centre_atom_indices
        n_atoms = len(self.s.atoms)

        if self.s.motif is not None:
            is_motif = self.s.motif.is_motif
            is_motif_related = self.s.motif.is_motif_related
        else:
            is_motif = np.zeros(n_atoms, dtype=int)
            is_motif_related = np.zeros(n_atoms, dtype=int)

        token_resolved = self.s.training.is_resolved[
            self.centre_atom_indices
        ]  # [N_token]
        is_motif_token = is_motif[centre]  # [N_token]
        is_motif_related_token = is_motif_related[centre]  # [N_token]

        restype = self.s.derived.cano_seq_resname[centre]
        restype_onehot = self.restype_onehot_encoded(restype)

        # TODO: this part is a bit ugly with the reshapes etc.
        motif_restype = restype_onehot * torch.from_numpy(
            (is_motif_related_token * token_resolved).astype(np.float32)
        ).unsqueeze(-1)  # [N_token, 32]

        motif_coordinate = self.s.token_centre_coords * is_motif_token.reshape(
            -1, 1
        )  # [N_token, 3]

        return MotifContext(
            is_motif=torch.from_numpy(is_motif.astype(bool)),
            is_motif_related=torch.from_numpy(is_motif_related.astype(bool)),
            is_motif_token=torch.from_numpy(is_motif_token.astype(bool)),
            is_motif_related_token=torch.from_numpy(
                is_motif_related_token.astype(bool)
            ),
            motif_restype=motif_restype.float(),
            motif_coordinate=torch.from_numpy(motif_coordinate.astype(np.float32)),
        )

    def get_coordinate_context(self) -> CoordinateContext:
        """Get coordinates features.

        Returns:
            CoordinateContext: Coordinates features.
        """
        return CoordinateContext(
            coordinate=torch.from_numpy(self.s.atoms.coord.astype(np.float32)),
            centre_coordinate=torch.from_numpy(
                self.s.token_centre_coords.astype(np.float32)
            ),
            backbone_coordinate=self.backbone_atoms_and_mask[0],
        )

    def get_basic_context(self, include_arrays: bool = True) -> BasicContext:
        """Get basic context information.

        Args:
            include_arrays: If True, include atom_array in the context.
                These are expensive to serialize and can be disabled for training
                performance. Defaults to True for backward compatibility.
        """
        return BasicContext(
            atom_array=self.s.atoms if include_arrays else None,
            token_array=None,
            pdb_id=self.pdb_id,
            bioassembly_dict_fpath=self.bioassembly_dict_fpath,
            n_asym=len(
                np.unique(self.s.entity_ids.asym_id_int[self.centre_atom_indices])
            ),
            n_token=self.s.grouping.n_tokens,
            n_atom=len(self.s.atoms),
            original_motif_mapping=self.original_motif_mapping,
        )

    def get_full_context(self, include_arrays: bool = True) -> FullContext:
        """Get all context required for design pipeline.

        Args:
            include_arrays: If True, include atom_array and token_array in BasicContext.
                These are expensive to serialize and can be disabled for training
                performance. Defaults to True for backward compatibility.
        """
        return FullContext(
            token=self.get_token_context(),
            reference=self.get_reference_conformer_context(),
            chain_perm=self.get_chain_perm_context(),
            # NOTE: atom_perm disabled - expensive, currently unused
            # atom_perm=self.get_atom_permutation_list(),
            mask=self.get_mask_context(),
            motif=self.get_motif_context(),
            index=self.get_index_context(),
            entity=self.get_entity_context(),
            coordinate=self.get_coordinate_context(),
            basic=self.get_basic_context(include_arrays=include_arrays),
        )

    def get_atom_permutation_list(
        self,
    ) -> AtomPermContext:
        """
        Generate info of permutations.

        Returns:
            List[List[int]]: a list of atom permutations.

        # TODO: add this back in.
        """
        atom_perm_list = []
        for i in self.s.atoms.res_perm:
            # Decode list[str] -> list[list[int]]
            atom_perm_list.append([int(j) for j in i.split("_")])

        # Atoms connected to different residue are fixed.
        # Bonds array: [[atom_idx_i, atom_idx_j, bond_type]]
        idx_i = self.s.atoms.bonds._bonds[:, 0]
        idx_j = self.s.atoms.bonds._bonds[:, 1]
        diff_mask = (
            self.s.reference.ref_space_uid[idx_i]
            != self.s.reference.ref_space_uid[idx_j]
        )
        inter_residue_bonds = self.s.atoms.bonds._bonds[diff_mask]
        fixed_atom_mask = np.isin(
            np.arange(len(self.s.atoms)),
            np.unique(inter_residue_bonds[:, :2]),
        )

        # Get fixed atom permutation for each residue.
        fixed_atom_perm_list = []
        res_starts = get_residue_starts(self.s.atoms, add_exclusive_stop=True)
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
        return AtomPermContext(perms=fixed_atom_perm_list)

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


def _extract_backbone_atoms(
    structure: AnnotatedStructure,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract backbone atom coordinates and masks for all tokens.

    Args:
        structure: AnnotatedStructure with grouping and atom data.

    Returns:
        backbone_coords: Backbone atom coordinates [n_token, 5, 3]
        backbone_mask: Mask indicating valid backbone atoms [n_token, 5]
    """
    g = structure.grouping
    atoms = structure.atoms
    centre_indices = structure.centre_atom_indices

    n_tokens = g.n_tokens
    backbone_coords = np.zeros((n_tokens, 5, 3), dtype=np.float32)
    backbone_mask = np.zeros((n_tokens, 5), dtype=np.float32)

    for token_idx in range(n_tokens):
        centre_idx = centre_indices[token_idx]
        mol_type = structure.type_masks.mol_type[centre_idx]
        res_name = atoms.res_name[centre_idx]
        is_standard = res_name in STD_RESIDUES

        # Ligands or non-standard residues: use centre atom only (no backbone)
        if mol_type == "ligand" or not is_standard:
            if structure.training.is_resolved[centre_idx]:
                backbone_coords[token_idx, 0] = atoms.coord[centre_idx]
                backbone_mask[token_idx, 0] = 1.0
            continue

        # Standard protein or nucleic acid — find this token's atoms
        token_atom_indices = np.where(g.atom_token_index == token_idx)[0]
        token_atom_names = atoms.atom_name[token_atom_indices]
        name_to_global = {
            name: token_atom_indices[i] for i, name in enumerate(token_atom_names)
        }

        backbone_names = BACKBONE_DEFS[mol_type]
        atom_coords = {}

        for bb_idx, atom_name in enumerate(backbone_names):
            global_idx = name_to_global.get(atom_name)
            if global_idx is not None:
                backbone_coords[token_idx, bb_idx] = atoms.coord[global_idx]
                resolved = structure.training.is_resolved[global_idx]
                backbone_mask[token_idx, bb_idx] = resolved
                if resolved:
                    atom_coords[atom_name] = atoms.coord[global_idx]

        # Compute virtual Cβ for proteins
        if mol_type == "protein" and {"N", "CA", "C"}.issubset(atom_coords):
            cb_coord = compute_virtual_cb(
                atom_coords["N"], atom_coords["CA"], atom_coords["C"]
            )
            backbone_coords[token_idx, 4] = cb_coord
            backbone_mask[token_idx, 4] = 1.0

    return torch.Tensor(backbone_coords), torch.Tensor(backbone_mask)
