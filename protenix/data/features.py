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
from typing import Annotated, Any, Callable, get_args, get_type_hints

import torch
from biotite.structure import AtomArray
from torch import Tensor

from protenix.data.padding_utils import pad_tensor
from protenix.data.tokenizer import TokenArray
from protenix.utils.typing import Bool, Float, Int, typecheck

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

            if not torch.is_tensor(value):
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

    token_bonds: Feature(
        Int[Tensor, "*batch n_token n_token"],
        type=FeatureType.TOKEN_PAIR,
        encoding=EncodingType.IDENTITY,
        description="Indicator for ligand-ligand and non-polymer bonds",
    )


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
    backbone_mask: Mask(
        Bool[Tensor, "*batch n_token 5"],
        type=FeatureType.TOKEN,
        description="Mask for all backbone atoms",
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
    backbone_coordinate: Feature(
        Float[Tensor, "*batch n_token 5 3"],
        type=FeatureType.TOKEN,
        description="Coordinates of backbone atoms",
    )
    resolution: float | list[float]


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


@dataclass
class FullContext:
    """Container for context required for Ligo's design pipeline."""

    token: TokenContext
    reference: ReferenceConformerContext
    chain_perm: ChainPermContext
    atom_perm: AtomPermContext
    mask: MaskContext
    motif: MotifContext
    index: IndexContext
    entity: EntityContext
    coordinate: CoordinateContext
    basic: BasicContext

    def pad(self, n_max_atoms: int, n_max_tokens: int) -> FullContext:
        """Pad all features to n_max_tokens/n_max_atoms."""
        current_n_token = self.basic.n_token
        current_n_atom = self.basic.n_atom

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

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert all Context fields to a nested dictionary.

        Returns:
            Nested dictionary with all fields from all sub-contexts.
        """
        result = {}
        for f in fields(self):
            result[f.name] = getattr(self, f.name).to_dict()
        return result

    @classmethod
    def from_dict(
        cls,
        d: dict[str, dict[str, Tensor]],
        resolution: float | list[float] = 0.0,
        basic_context: BasicContext | None = None,
    ) -> FullContext:
        """Create FullContext from nested dictionary of tensors.

        This is the inverse of `to_dict()`. Since `to_dict()` only saves tensor
        fields, non-tensor fields (BasicContext, resolution) must be provided
        separately or will use defaults.

        Args:
            d: Nested dictionary mapping context names to dicts of tensors.
            resolution: Resolution value for CoordinateContext.
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
        atom_perm = AtomPermContext.from_dict(d.get("atom_perm", {}))
        mask = MaskContext.from_dict(d.get("mask", {}))
        motif = MotifContext.from_dict(d.get("motif", {}))
        index = IndexContext.from_dict(d.get("index", {}))
        entity = EntityContext.from_dict(d.get("entity", {}))
        coordinate = CoordinateContext.from_dict(
            d.get("coordinate", {}), resolution=resolution
        )

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

        return cls(
            token=token,
            reference=reference,
            chain_perm=chain_perm,
            atom_perm=atom_perm,
            mask=mask,
            motif=motif,
            index=index,
            entity=entity,
            coordinate=coordinate,
            basic=basic_context,
        )
