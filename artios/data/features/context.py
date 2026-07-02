"""Typed feature containers for all-atom structure modeling.

This module provides dataclass-based containers for features with runtime
shape and dtype validation via jaxtyping + beartype.

Example usage:
    >>> features = AllAtomFeatures.from_featurizer(featurizer)
    >>> features = features.to("cuda")
    >>> features.token.restype  # IDE autocomplete works
    >>> features.pad(n_token_pad=10, n_atom_pad=20)

Feature metadata can be accessed via:
    >>> TokenFeatures.get_field_meta("restype")
    FeatureMeta(encoding=Encoding.ONE_HOT, ...)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from enum import Enum, auto
from functools import partial
from typing import Annotated, Any, Callable, Optional, get_args, get_type_hints

import torch
from biotite.structure import AtomArray
from torch import Tensor

from artios.data.collate.padding import pad_tensor
from artios.data.structure.annotation import MotifMappingEntry
from artios.data.structure.tokenizer import TokenArray
from artios.utils.typing import Bool, Float, Int, typecheck

logger = logging.getLogger(__name__)


class FeatureRole(Enum):
    """Role of the feature in the model."""

    INPUT = auto()
    LABEL = auto()
    MASK = auto()  # masks can be input into model as well e.g. ref_mask
    AUXILIARY = auto()


class FeatureType(Enum):
    """Semantic categorization of features by their indexing dimension."""

    TOKEN = "TOKEN"
    TOKEN_PAIR = "TOKEN_PAIR"
    ATOM = "ATOM"
    ATOM_PAIR = "ATOM_PAIR"


class EncodingType(Enum):
    ONE_HOT = "one-hot"
    RBF = "rbf"
    FOURIER = "fourier"
    IDENTITY = "identity"


@dataclass(frozen=True)
class FeatureMeta:
    """Metadata for a tensor feature field.

    Attributes:
        encoding: How the values are encoded (one-hot, continuous, etc.)
        type: Feature type (TOKEN, ATOM, etc.) for padding inference
        description: Human-readable description
    """

    encoding: EncodingType | None = None
    type: FeatureType | None = None
    role: FeatureRole = FeatureRole.INPUT
    description: str = ""


def Feature(
    jaxtype,
    *,
    type: FeatureType,
    encoding: EncodingType | None = None,
    description: str = "",
):
    """Annotate a jaxtyped tensor as a model input feature."""
    return Annotated[
        jaxtype, FeatureMeta(encoding, type, FeatureRole.INPUT, description)
    ]


def Label(
    jaxtype,
    *,
    type: FeatureType,
    encoding: EncodingType | None = None,
    description: str = "",
):
    """Annotate a jaxtyped tensor as a supervision label."""
    return Annotated[
        jaxtype, FeatureMeta(encoding, type, FeatureRole.LABEL, description)
    ]


def Mask(
    jaxtype,
    *,
    type: FeatureType,
    encoding: EncodingType | None = None,
    description: str = "",
):
    """Annotate a jaxtyped tensor as a mask for loss/attention."""
    return Annotated[
        jaxtype, FeatureMeta(encoding, type, FeatureRole.MASK, description)
    ]


def Auxiliary(
    jaxtype,
    *,
    type: FeatureType,
    encoding: EncodingType | None = None,
    description: str = "",
):
    """Annotate a jaxtyped tensor as auxiliary data (not directly used by model)."""
    return Annotated[
        jaxtype, FeatureMeta(encoding, type, FeatureRole.AUXILIARY, description)
    ]


@dataclass
class Context:
    """Base class with common utilities for feature containers."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary of field values."""
        return dict(asdict(self).items())

    @classmethod
    def from_dict(cls, d: dict[str, Any], **extra_kwargs) -> Context:
        """Create instance from dictionary of tensors.

        Args:
            d: Dictionary mapping field names to tensors.
            **extra_kwargs: Additional non-tensor fields to set.

        Returns:
            New instance with fields populated from dict.
        """
        kwargs = {}
        for f in fields(cls):
            if f.name in d:
                kwargs[f.name] = d[f.name]
            elif f.name in extra_kwargs:
                kwargs[f.name] = extra_kwargs[f.name]
        return cls(**kwargs)

    def to(self, device: torch.device, non_blocking: bool = False) -> Context:
        """Move all tensor fields to the specified device.

        Args:
            device: Target device (e.g., torch.device("cuda")).
            non_blocking: If True, perform async transfer for pinned memory.

        Returns:
            New instance with tensors moved to device.
        """
        moved_fields = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if torch.is_tensor(value):
                moved_fields[f.name] = value.to(device, non_blocking=non_blocking)
            else:
                moved_fields[f.name] = value
        return type(self)(**moved_fields)

    def pad(
        self,
        pad_fns: dict[FeatureType, Callable[[Tensor], Tensor]],
    ) -> Context:
        """Pad all tensor fields using per-field feature type.

        Args:
            pad_fns: Dict mapping FeatureType to padding functions.
                Each function takes a tensor and returns the padded tensor.

        Returns:
            New instance with padded tensors.
        """
        type_hints = get_type_hints(type(self), include_extras=True)

        padded_fields = {}
        for f in fields(self):
            value = getattr(self, f.name)

            if value is None or not torch.is_tensor(value):
                padded_fields[f.name] = value
                continue

            ann = type_hints[f.name]
            feature_type = get_args(ann)[1].type
            padded_fields[f.name] = pad_fns[feature_type](value)

        return type(self)(**padded_fields)


@typecheck
@dataclass
class TokenContext(Context):
    """Token-level features from AlphaFold3 SI Chapter 2.8.

    All tensors have shape [*batch, n_token, ...] where *batch is optional batch dims.
    """

    token_index: Feature(Int[Tensor, "*batch n_token"], type=FeatureType.TOKEN)
    residue_index: Feature(Int[Tensor, "*batch n_token"], type=FeatureType.TOKEN)
    asym_id: Feature(Int[Tensor, "*batch n_token"], type=FeatureType.TOKEN)
    entity_id: Feature(Int[Tensor, "*batch n_token"], type=FeatureType.TOKEN)
    sym_id: Feature(Int[Tensor, "*batch n_token"], type=FeatureType.TOKEN)
    restype: Feature(
        Int[Tensor, "*batch n_token 32"],
        type=FeatureType.TOKEN,
        encoding=EncodingType.ONE_HOT,
    )

    # NOTE: Temporarily optional - expensive to compute, only needed for MPNN
    token_bonds: Optional[
        Feature(
            Int[Tensor, "*batch n_token n_token"],
            type=FeatureType.TOKEN_PAIR,
            encoding=EncodingType.IDENTITY,
            description="Indicator for ligand-ligand and non-polymer bonds",
        )
    ] = None


@typecheck
@dataclass
class ReferenceConformerContext(Context):
    """Reference conformer features from AlphaFold3 SI Chapter 2.8."""

    ref_pos: Feature(
        Float[Tensor, "*batch n_atom 3"],
        type=FeatureType.ATOM,
        encoding=EncodingType.IDENTITY,
        description="Reference conformer positions",
    )
    ref_mask: Mask(
        Int[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        encoding=EncodingType.IDENTITY,
        description="Mask indicating atom slots used in conformer",
    )
    ref_element: Feature(
        Int[Tensor, "*batch n_atom 128"],
        type=FeatureType.ATOM,
        encoding=EncodingType.ONE_HOT,
    )
    ref_charge: Feature(
        Int[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        encoding=EncodingType.IDENTITY,
    )
    ref_atom_name_chars: Feature(
        Int[Tensor, "*batch n_atom 4 64"],
        type=FeatureType.ATOM,
        encoding=EncodingType.ONE_HOT,
    )
    ref_space_uid: Auxiliary(  # TODO: i think this is an index
        Int[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        description="ID for each unique conformer",
    )


@typecheck
@dataclass
class ChainPermContext(Context):
    """Chain permutation features for multi-chain alignment.

    Used for permutation-invariant training with symmetric chains.
    """

    mol_id: Auxiliary(Int[Tensor, "*batch n_atom"], type=FeatureType.ATOM)
    mol_atom_index: Auxiliary(Int[Tensor, "*batch n_atom"], type=FeatureType.ATOM)
    entity_mol_id: Auxiliary(Int[Tensor, "*batch n_atom"], type=FeatureType.ATOM)


@dataclass
class AtomPermContext(Context):
    """Atom permutation features for symmetric permutation.

    TODO: feature explanation
    """

    perms: list[list[int]] = field(default_factory=list)


@typecheck
@dataclass
class IndexContext(Context):
    """Atom-to-token index mappings and global features.

    Contains auxiliary indexing information for relating atoms to tokens.
    """

    atom_to_token_idx: Auxiliary(
        Int[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        description="Token index for each atom.",
    )
    atom_to_tokatom_idx: Auxiliary(
        Int[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        description="Token atom (which atom within token) index for each atom",
    )
    atom_asym_id: Auxiliary(
        Int[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        description="Chain ID at atom level (reflects motif unindexing)",
    )


@typecheck
@dataclass
class EntityContext(Context):
    """Molecule type indicators at atom and token level.

    Boolean masks indicating which molecule type each atom/token belongs to.
    """

    # Atom-level indicators
    is_protein: Mask(Bool[Tensor, "*batch n_atom"], type=FeatureType.ATOM)
    is_ligand: Mask(Bool[Tensor, "*batch n_atom"], type=FeatureType.ATOM)
    is_dna: Mask(Bool[Tensor, "*batch n_atom"], type=FeatureType.ATOM)
    is_rna: Mask(Bool[Tensor, "*batch n_atom"], type=FeatureType.ATOM)

    # Token-level indicators
    is_protein_token: Mask(Bool[Tensor, "*batch n_token"], type=FeatureType.TOKEN)
    is_ligand_token: Mask(Bool[Tensor, "*batch n_token"], type=FeatureType.TOKEN)
    is_dna_token: Mask(Bool[Tensor, "*batch n_token"], type=FeatureType.TOKEN)
    is_rna_token: Mask(Bool[Tensor, "*batch n_token"], type=FeatureType.TOKEN)


@typecheck
@dataclass
class MaskContext(Context):
    """Mask features for loss computation and attention.

    Contains various masks at both token and atom level.
    """

    centre_atom_mask: Mask(
        Bool[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        description="Atom is centre atom of the token (0 for non-centre atoms)",
    )
    atom_mask: Mask(
        Bool[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        description="Atom is not padding (for attention masking)",
    )
    is_resolved: Mask(
        Bool[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        description="Atom is resolved in structure.",
    )
    centre_atom_is_resolved: Mask(
        Bool[Tensor, "*batch n_token"],
        type=FeatureType.TOKEN,
        description="Centre atom is resolved in structure.",
    )
    atom_loss_mask: Mask(
        Bool[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        description="Atom is resolved in structure and is in the token with a resolved centre atom.",
    )
    token_mask: Mask(
        Bool[Tensor, "*batch n_token"],
        type=FeatureType.TOKEN,
        description="Token is not padding mask (all true before pad)",
    )
    # NOTE: Temporarily optional - expensive to compute, only needed for MPNN
    backbone_mask: Mask(
        Bool[Tensor, "*batch n_token 5"],
        type=FeatureType.TOKEN,
        description="Mask for all backbone atoms",
    ) = None


@typecheck
@dataclass
class MotifContext(Context):
    """Motif features for motif scaffolding."""

    # Atom-level
    is_motif: Mask(
        Bool[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        description="Is motif atom.",
    )
    is_motif_related: Mask(
        Bool[Tensor, "*batch n_atom"],
        type=FeatureType.ATOM,
        description="Is a conformer (e.g. sidechain) with at least one atom in the motif.",
    )

    # Token-level
    is_motif_token: Mask(
        Bool[Tensor, "*batch n_token"],
        type=FeatureType.TOKEN,
        description="Is motif token (has motif coordinates)",
    )
    is_motif_related_token: Mask(
        Bool[Tensor, "*batch n_token"],
        type=FeatureType.TOKEN,
        description="Is a token related to a motif (e.g. sidechain, which are tokenized per-atom)",
    )
    motif_restype: Feature(
        Float[Tensor, "*batch n_token 32"],
        type=FeatureType.TOKEN,
        description="Motif residue type",
    )
    motif_coordinate: Feature(
        Float[Tensor, "*batch n_token 3"],
        type=FeatureType.TOKEN,
        encoding=EncodingType.IDENTITY,
        description="Motif coordinates",
    )
    motif_depth_norm: Feature(
        Float[Tensor, "*batch n_token"],
        type=FeatureType.TOKEN,
        encoding=EncodingType.IDENTITY,
        description=(
            "Normalised motif depth ||CoM(motif) - CoM(non-motif)|| / Rg_complex, "
            "broadcast across all tokens. -1.0 = null / unconditioned."
        ),
    )


@typecheck
@dataclass
class RasaContext(Context):
    """Relative solvent accessibility (RASA) conditioning features."""

    rasa: Feature(
        Float[Tensor, "*batch n_token"],
        type=FeatureType.TOKEN,
        encoding=EncodingType.IDENTITY,
        description=(
            "Per-token ligand fractional exposure SASA(complex) / SASA(isolated) "
            "in [0, 1]; -1.0 = null / unconditioned. Non-ligand tokens are -1.0."
        ),
    )


@typecheck
@dataclass
class FixedResidueContext(Context):
    """Identity-fixed residue information for sequence design.

    Positions whose amino-acid identity is constrained during
    sequence design.
    """

    fixed_mask: Mask(
        Bool[Tensor, "*batch n_token"],
        type=FeatureType.TOKEN,
        description="True at positions whose residue identity is fixed",
    )
    fixed_restype: Feature(
        Float[Tensor, "*batch n_token 32"],
        type=FeatureType.TOKEN,
        encoding=EncodingType.ONE_HOT,
        description="One-hot restype at fixed positions, zero elsewhere",
    )


@typecheck
@dataclass
class MSAContext(Context):
    """MSA features for one reference chain, token-aligned to the cropped structure.

    Shape convention: n_token is the leading non-batch axis so that the
    existing FeatureType.TOKEN pad_fn (pads the last non-batch dim by
    n_token_pad) applies uniformly to all fields. The MSA dataclass
    (artios/data/msa/sampler.py) stores the transposed MSA-native
    ``[n_msa_plus_one, l_chain]`` layout (one query row at index 0 plus
    ``n_msa`` homolog rows); ``ContextGenerator._build_msa_context``
    transposes and scatters chain columns into their n_token positions.

    Vocabulary (same as ``artios.common.constants.STD_RESIDUES_AND_GAP``):
      - 0..19: canonical amino acids, alphabetical by 3-letter code
               (ALA=0, ARG=1, ..., VAL=19).
      - 20: UNK (X / B / Z / J / U / O / any non-canonical character).
      - 21..30: RNA/DNA slots in the shared vocab — never appear in
               protein MSAs.
      - 31: gap ('-').

    Non-reference-chain tokens are gap-filled (31); ``msa_chain_mask``
    marks which tokens carry real MSA data.
    """

    tokens: Feature(
        Int[Tensor, "*batch n_token n_msa_plus_one"],
        type=FeatureType.TOKEN,
        encoding=EncodingType.IDENTITY,
        description=(
            "Integer token IDs in [0, 31] from STD_RESIDUES_AND_GAP — NOT "
            "three-letter codes or strings. Axis layout: "
            "n_token × n_msa_plus_one. Column 0 is the query row; "
            "columns 1..n_msa are homologs (n_msa_plus_one = n_msa + 1). "
            "Rows (n_token axis) outside the reference chain are gap-filled "
            "with 31. Encoding: 0..19 canonical AAs (ALA..VAL alphabetical), "
            "20 UNK, 31 gap."
        ),
    )
    gap_mask: Mask(
        Bool[Tensor, "*batch n_token n_msa_plus_one"],
        type=FeatureType.TOKEN,
        description="True where tokens == 31 (gap). Materialised for convenience.",
    )
    msa_chain_mask: Mask(
        Bool[Tensor, "*batch n_token"],
        type=FeatureType.TOKEN,
        description="True on tokens belonging to the reference MSA chain.",
    )
    msa_chain_asym_id: int | list[int]  # becomes list[int] after batching


@typecheck
@dataclass
class CoordinateContext(Context):
    """Coordinate features."""

    coordinate: Feature(
        Float[Tensor, "*batch n_atom 3"],
        type=FeatureType.ATOM,
        description="Coordinates of full structure",
    )
    centre_coordinate: Feature(
        Float[Tensor, "*batch n_token 3"],
        type=FeatureType.TOKEN,
        description="Coordinates of centre atoms",
    )
    # NOTE: Temporarily optional - expensive to compute, only needed for MPNN
    backbone_coordinate: Feature(
        Float[Tensor, "*batch n_token 5 3"],
        type=FeatureType.TOKEN,
        description="Coordinates of backbone atoms",
    ) = None


@typecheck
@dataclass
class BasicContext(Context):
    """Contains basic metadata about the sample.

    Note: This context contains non-tensor fields that become lists when batched.
    The atom_array and token_array fields can be None when loading from serialized
    data (e.g., from_dict).
    """

    atom_array: AtomArray | list[AtomArray] | list[None] | None
    token_array: TokenArray | list[TokenArray] | list[None] | None

    pdb_id: str | list[str]
    bioassembly_dict_fpath: str | list[str]
    n_asym: int | list[int]
    n_token: int | list[int]
    n_atom: int | list[int]

    original_motif_mapping: (
        tuple[MotifMappingEntry, ...]
        | list[tuple[MotifMappingEntry, ...]]
        | list[None]
        | None
    ) = None


@dataclass
class FullContext:
    """Container for context required for Ligo's design pipeline."""

    token: TokenContext
    reference: ReferenceConformerContext
    chain_perm: ChainPermContext
    mask: MaskContext
    motif: MotifContext
    index: IndexContext
    entity: EntityContext
    coordinate: CoordinateContext
    basic: BasicContext
    rasa: RasaContext
    # NOTE: Temporarily optional - expensive to compute, currently unused
    atom_perm: AtomPermContext = field(default_factory=AtomPermContext)
    fixed_residues: FixedResidueContext | None = None
    msa: MSAContext | None = None

    def pad(self, n_max_atoms: int, n_max_tokens: int) -> FullContext:
        """Pad all features to n_max_tokens/n_max_atoms."""
        # Padding is done before batching, so these are always int
        current_n_token = self.basic.n_token
        current_n_atom = self.basic.n_atom
        assert isinstance(current_n_token, int)
        assert isinstance(current_n_atom, int)

        assert n_max_tokens >= current_n_token
        assert n_max_atoms >= current_n_atom

        n_token_pad = n_max_tokens - current_n_token
        n_atom_pad = n_max_atoms - current_n_atom

        pad_fns = {
            FeatureType.TOKEN: partial(pad_tensor, pad_sizes=(n_token_pad,)),
            FeatureType.TOKEN_PAIR: partial(
                pad_tensor, pad_sizes=(n_token_pad, n_token_pad)
            ),
            FeatureType.ATOM: partial(pad_tensor, pad_sizes=(n_atom_pad,)),
            FeatureType.ATOM_PAIR: partial(
                pad_tensor, pad_sizes=(n_atom_pad, n_atom_pad)
            ),
        }

        kwargs = {}
        for f in fields(self):
            fg = getattr(self, f.name)
            if isinstance(fg, Context):
                kwargs[f.name] = fg.pad(pad_fns)
            else:
                kwargs[f.name] = fg

        return type(self)(**kwargs)

    def to(self, device: torch.device, non_blocking: bool = False) -> FullContext:
        """Move all tensor fields in all sub-contexts to the specified device.

        Args:
            device: Target device (e.g., torch.device("cuda")).
            non_blocking: If True, perform async transfer for pinned memory.

        Returns:
            New FullContext with all tensors moved to device.
        """
        kwargs = {}
        for f in fields(self):
            fg = getattr(self, f.name)
            if isinstance(fg, Context):
                kwargs[f.name] = fg.to(device, non_blocking=non_blocking)
            else:
                kwargs[f.name] = fg
        return type(self)(**kwargs)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert all Context fields to a nested dictionary.

        Returns:
            Nested dictionary with all fields from all sub-contexts.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, Context):
                result[f.name] = value.to_dict()
            else:
                result[f.name] = value
        return result

    def to_unconditional(self) -> FullContext:
        """Create unconditional version with motif and ligand conditioning zeroed.
        Used for classifier-free guidance.

        Returns:
            New FullContext with motif and ligand features zeroed.
        """
        # uncond_motif = MotifContext(
        #     is_motif=torch.zeros_like(self.motif.is_motif),
        #     is_motif_related=torch.zeros_like(self.motif.is_motif_related),
        #     is_motif_token=torch.zeros_like(self.motif.is_motif_token),
        #     is_motif_related_token=torch.zeros_like(self.motif.is_motif_related_token),
        #     motif_restype=torch.zeros_like(self.motif.motif_restype),
        #     motif_coordinate=torch.zeros_like(self.motif.motif_coordinate),
        # )

        uncond_entity = EntityContext(
            is_protein=self.entity.is_protein,
            is_ligand=torch.zeros_like(self.entity.is_ligand),
            is_dna=self.entity.is_dna,
            is_rna=self.entity.is_rna,
            is_protein_token=self.entity.is_protein_token,
            is_ligand_token=torch.zeros_like(self.entity.is_ligand_token),
            is_dna_token=self.entity.is_dna_token,
            is_rna_token=self.entity.is_rna_token,
        )

        return FullContext(
            token=self.token,
            reference=self.reference,
            chain_perm=self.chain_perm,
            mask=self.mask,
            motif=self.motif,
            index=self.index,
            entity=uncond_entity,
            coordinate=self.coordinate,
            basic=self.basic,
            rasa=self.rasa,
            fixed_residues=self.fixed_residues,
            msa=self.msa,
        )

    def without_ligand(self) -> FullContext:
        """Create a context with all ligand influence removed.

        Unlike to_unconditional() which only zeros entity flags, this method
        also removes ligand tokens from token_mask (excluding them from the
        neighbor graph and attention) and zeroes all ligand-related features
        in motif and coordinate contexts.

        Used for classifier-free guidance in Potts sampling.
        """
        is_lig = self.entity.is_ligand
        is_lig_tok = self.entity.is_ligand_token

        not_lig = ~is_lig
        not_lig_tok = ~is_lig_tok

        new_mask = MaskContext(
            centre_atom_mask=self.mask.centre_atom_mask & not_lig,
            atom_mask=self.mask.atom_mask & not_lig,
            is_resolved=self.mask.is_resolved,
            centre_atom_is_resolved=self.mask.centre_atom_is_resolved,
            atom_loss_mask=self.mask.atom_loss_mask,
            token_mask=self.mask.token_mask & not_lig_tok,
            backbone_mask=self.mask.backbone_mask,
        )

        new_entity = EntityContext(
            is_protein=self.entity.is_protein,
            is_ligand=torch.zeros_like(is_lig),
            is_dna=self.entity.is_dna,
            is_rna=self.entity.is_rna,
            is_protein_token=self.entity.is_protein_token,
            is_ligand_token=torch.zeros_like(is_lig_tok),
            is_dna_token=self.entity.is_dna_token,
            is_rna_token=self.entity.is_rna_token,
        )

        new_motif = MotifContext(
            is_motif=self.motif.is_motif & not_lig,
            is_motif_related=self.motif.is_motif_related & not_lig,
            is_motif_token=self.motif.is_motif_token & not_lig_tok,
            is_motif_related_token=self.motif.is_motif_related_token & not_lig_tok,
            motif_restype=self.motif.motif_restype * not_lig_tok.unsqueeze(-1),
            motif_coordinate=self.motif.motif_coordinate * not_lig_tok.unsqueeze(-1),
            motif_depth_norm=self.motif.motif_depth_norm,
        )

        new_coordinate = CoordinateContext(
            coordinate=self.coordinate.coordinate * not_lig.unsqueeze(-1),
            centre_coordinate=self.coordinate.centre_coordinate
            * not_lig_tok.unsqueeze(-1),
            backbone_coordinate=self.coordinate.backbone_coordinate,
        )

        new_rasa = RasaContext(
            rasa=torch.where(
                not_lig_tok,
                self.rasa.rasa,
                torch.full_like(self.rasa.rasa, -1.0),
            ),
        )

        return FullContext(
            token=self.token,
            reference=self.reference,
            chain_perm=self.chain_perm,
            mask=new_mask,
            motif=new_motif,
            index=self.index,
            entity=new_entity,
            coordinate=new_coordinate,
            basic=self.basic,
            rasa=new_rasa,
            fixed_residues=self.fixed_residues,
            msa=self.msa,
        )

    @classmethod
    def from_dict(
        cls,
        d: dict[str, dict[str, Tensor]],
        basic_context: BasicContext | None = None,
    ) -> FullContext:
        """Create FullContext from nested dictionary of tensors.

        This is the inverse of `to_dict()`. Since `to_dict()` only saves tensor
        fields, BasicContext must be provided separately or will use defaults.

        Args:
            d: Nested dictionary mapping context names to dicts of tensors.
            basic_context: Optional BasicContext. If None, creates placeholder.

        Returns:
            New FullContext instance.

        Example:
            >>> data = torch.load("batch.pt")
            >>> context = FullContext.from_dict(data)
        """
        # Create sub-contexts from their dictionaries
        token = TokenContext.from_dict(d.get("token", {}))
        reference = ReferenceConformerContext.from_dict(d.get("reference", {}))
        chain_perm = ChainPermContext.from_dict(d.get("chain_perm", {}))
        # atom_perm = AtomPermContext.from_dict(d.get("atom_perm", {}))
        mask = MaskContext.from_dict(d.get("mask", {}))
        motif = MotifContext.from_dict(d.get("motif", {}))
        index = IndexContext.from_dict(d.get("index", {}))
        entity = EntityContext.from_dict(d.get("entity", {}))
        coordinate = CoordinateContext.from_dict(d.get("coordinate", {}))
        rasa = RasaContext.from_dict(d.get("rasa", {}))

        # BasicContext requires non-tensor fields - use placeholder if not provided
        if basic_context is None:
            # Infer dimensions from tensor shapes
            n_token = (
                token.token_index.shape[-1] if token.token_index is not None else 0
            )
            n_atom = mask.atom_mask.shape[-1] if mask.atom_mask is not None else 0
            batch_size = (
                token.token_index.shape[0] if token.token_index.dim() > 1 else 1
            )

            basic_context = BasicContext(
                atom_array=[None] * batch_size,  # type: ignore
                token_array=[None] * batch_size,  # type: ignore
                pdb_id=["unknown"] * batch_size,
                bioassembly_dict_fpath=[""] * batch_size,
                n_asym=[1] * batch_size,
                n_token=[int(n_token)] * batch_size,
                n_atom=[int(n_atom)] * batch_size,
            )

        # Optional: FixedResidueContext
        fixed_residues = None
        if "fixed_residues" in d:
            fixed_residues = FixedResidueContext.from_dict(d["fixed_residues"])

        # Optional: MSAContext
        msa = None
        if "msa" in d:
            msa = MSAContext.from_dict(d["msa"])

        return cls(
            token=token,
            reference=reference,
            chain_perm=chain_perm,
            # atom_perm=atom_perm,
            mask=mask,
            motif=motif,
            index=index,
            entity=entity,
            coordinate=coordinate,
            basic=basic_context,
            rasa=rasa,
            fixed_residues=fixed_residues,
            msa=msa,
        )
