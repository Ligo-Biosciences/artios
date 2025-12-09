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
from dataclasses import asdict, dataclass, fields
from enum import Enum, auto
from functools import partial
from typing import Annotated, Callable

import torch
from torch import Tensor

from protenix.model.utils import pad_tensor
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
class FeatureGroup:
    """Base class with common utilities for feature containers."""

    def to_dict(self) -> dict[str, Tensor]:
        """Convert to flat dictionary of tensors."""
        return {k: v for k, v in asdict(self).items() if torch.is_tensor(v)}

    def pad(
        self,
        pad_fns: dict[FeatureType, Callable[[Tensor], Tensor]],
    ) -> FeatureGroup:
        """Pad all tensor fields using per-field feature type.

        Args:
            pad_fns: Dict mapping FeatureType to padding functions.
                Each function takes a tensor and returns the padded tensor.

        Returns:
            New instance with padded tensors.
        """
        padded_fields = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                padded_fields[field.name] = pad_fns[field.type](value)
            else:
                padded_fields[field.name] = value
        return type(self)(**padded_fields)


@typecheck
@dataclass
class TokenFeatures(FeatureGroup):
    """Token-level features from AlphaFold3 SI Chapter 2.8.

    All tensors have shape [n_token, ...] where n_token is the number of tokens.
    """

    token_index: Feature(Int[Tensor, "n_token"], type=FeatureType.TOKEN)
    residue_index: Feature(Int[Tensor, "n_token"], type=FeatureType.TOKEN)
    asym_id: Feature(Int[Tensor, "n_token"], type=FeatureType.TOKEN)
    entity_id: Feature(Int[Tensor, "n_token"], encoding=EncodingType.IDENTITY)
    sym_id: Feature(Int[Tensor, "n_token"], encoding=EncodingType.IDENTITY)
    restype: Feature(Int[Tensor, "n_token 32"], encoding=EncodingType.ONE_HOT)

    token_bonds: Feature(
        Int[Tensor, "n_token n_token"],
        type=FeatureType.TOKEN_PAIR,
        encoding=EncodingType.IDENTITY,
        description="Indicator for ligand-ligand and non-polymer bonds",
    )

    token_mask: Mask(
        Bool[Tensor, "n_token"],
        type=FeatureType.TOKEN,
        description="Token is not padding mask (all true before pad)",
    )


@typecheck
@dataclass
class ReferenceFeatures(FeatureGroup):
    """Reference conformer features from AlphaFold3 SI Chapter 2.8."""

    ref_pos: Feature(
        Float[Tensor, "n_atom 3"],
        type=FeatureType.ATOM,
        encoding=EncodingType.IDENTITY,
        description="Reference conformer positions",
    )
    ref_mask: Mask(
        Int[Tensor, "n_atom"],
        type=FeatureType.ATOM,
        encoding=EncodingType.IDENTITY,
        description="Mask indicating atom slots used in conformer",
    )
    ref_element: Feature(
        Int[Tensor, "n_atom 128"],
        type=FeatureType.ATOM,
        encoding=EncodingType.ONE_HOT,
    )
    ref_charge: Feature(
        Int[Tensor, "n_atom"],
        type=FeatureType.ATOM,
        encoding=EncodingType.IDENTITY,
    )
    ref_atom_name_chars: Feature(
        Int[Tensor, "n_atom 4 64"],
        type=FeatureType.ATOM,
        encoding=EncodingType.ONE_HOT,
    )
    ref_space_uid: Auxiliary(
        Int[Tensor, "n_atom"],
        type=FeatureType.ATOM,
        description="ID for each unique conformer",
    )


@typecheck
@dataclass
class ChainPermFeatures(FeatureGroup):
    """Chain permutation features for multi-chain alignment.

    Used for permutation-invariant training with symmetric chains.
    """

    mol_id: Int[Tensor, "n_atom"]
    mol_atom_index: Int[Tensor, "n_atom"]
    entity_mol_id: Int[Tensor, "n_atom"]


class AtomPermFeatures(FeatureGroup):
    """Atom permutation features for symmetric permutation

    TODO: feature explanation
    """

    perms: list[list[int]]


@typecheck
@dataclass
class ExtraFeatures(FeatureGroup):
    """Additional features not in AlphaFold3 SI Table 5.

    Contains atom-to-token mappings and molecule type indicators.
    """

    # Atom-level
    atom_to_token_idx: Auxiliary(
        Int[Tensor, "n_atom"],
        type=FeatureType.ATOM,
        description="Token index for each atom.",
    )
    atom_to_tokatom_idx: Auxiliary(
        Int[Tensor, "n_atom"],
        type=FeatureType.ATOM,
        description="Token atom (which atom within token) index for each atom",
    )

    # Indicators
    is_protein: Mask(Bool[Tensor, "n_atom"], type=FeatureType.ATOM)
    is_ligand: Mask(Bool[Tensor, "n_atom"], type=FeatureType.ATOM)
    is_dna: Mask(Bool[Tensor, "n_atom"], type=FeatureType.ATOM)
    is_rna: Mask(Bool[Tensor, "n_atom"], type=FeatureType.ATOM)

    # Token-level
    is_protein_token: Mask(Bool[Tensor, "n_token"], type=FeatureType.TOKEN)
    is_ligand_token: Mask(Bool[Tensor, "n_token"], type=FeatureType.TOKEN)
    is_dna_token: Mask(Bool[Tensor, "n_token"], type=FeatureType.TOKEN)
    is_rna_token: Mask(Bool[Tensor, "n_token"], type=FeatureType.TOKEN)

    # Global
    resolution: float


@typecheck
@dataclass
class MaskFeatures(FeatureGroup):
    """Mask features for loss computation and attention.

    Contains various masks at both token and atom level.
    """

    # Atom-level [n_atom]
    pae_rep_atom_mask: Bool[Tensor, "n_atom"]
    plddt_m_rep_atom_mask: Bool[Tensor, "n_atom"]
    modified_res_mask: Bool[Tensor, "n_atom"]
    atom_mask: Bool[Tensor, "n_atom"]


@typecheck
@dataclass
class MPNNFeatures(FeatureGroup):
    """Features for MPNN-style sequence design.

    Contains backbone coordinates and masks for protein design.
    """

    mpnn_centre_coordinate: Feature(Float[Tensor, "n_token 3"], type=FeatureType.TOKEN)
    mpnn_centre_atom_mask: Feature(Bool[Tensor, "n_token"], type=FeatureType.TOKEN)
    mpnn_backbone_coords: Feature(
        Float[Tensor, "n_token 5 3"], type=FeatureType.TOKEN
    )  # N, CA, C, O, CB
    mpnn_backbone_mask: Feature(Bool[Tensor, "n_token 3"], type=FeatureType.TOKEN)


@typecheck
@dataclass
class MotifFeatures(FeatureGroup):
    """Motif features for motif scaffolding."""

    # Atom-level
    is_motif: Mask(
        Bool[Tensor, "n_atom"],
        type=FeatureType.ATOM,
        description="Is motif atom.",
    )
    is_motif_related: Mask(
        Bool[Tensor, "n_atom"],
        type=FeatureType.ATOM,
        description="Is a conformer (e.g. sidechain) with at least one atom in the motif.",
    )

    # Token-level
    is_motif_token: Mask(
        Int[Tensor, "n_token"],
        type=FeatureType.TOKEN,
        description="Is motif token (has motif coordinates)",
    )
    is_motif_related_token: Mask(
        Int[Tensor, "n_token"],
        type=FeatureType.TOKEN,
        description="Is a token related to a motif (e.g. sidechain, which are tokenized per-atom)",
    )
    motif_restype: Feature(
        Float[Tensor, "n_token 32"],
        type=FeatureType.TOKEN,
        description="Motif residue type",
    )
    motif_coordinate: Feature(
        Float[Tensor, "n_token 3"],
        type=FeatureType.TOKEN,
        encoding=EncodingType.IDENTITY,
        description="Motif coordinates",
    )


@typecheck
@dataclass
class StructureLabels(FeatureGroup):
    """Ground truth labels for structure prediction training.

    Contains coordinates and masks for computing losses.
    """

    # Atom-level
    coordinate: Label(
        Float[Tensor, "n_atom 3"],
        type=FeatureType.ATOM,
        encoding=EncodingType.IDENTITY,
    )
    coordinate_mask: Mask(
        Bool[Tensor, "n_atom"],
        type=FeatureType.ATOM,
        encoding=EncodingType.IDENTITY,
    )
    # TODO: is atom loss mask even necessary?
    atom_loss_mask: Mask(
        Bool[Tensor, "n_atom"],
        type=FeatureType.ATOM,
    )

    # Token-level
    centre_coordinate: Label(Float[Tensor, "n_token 3"], type=FeatureType.TOKEN)
    has_centre_atom: Mask(Bool[Tensor, "n_token"], type=FeatureType.TOKEN)
    token_loss_mask: Mask(Bool[Tensor, "n_token"], type=FeatureType.TOKEN)
    dense_to_flat_atom_mask: Mask(
        Bool[Tensor, "n_token n_dense_atoms"], type=FeatureType.TOKEN
    )


@dataclass
class AllAtomFeatures:
    """Container for all features needed for all-atom structure prediction.

    This is the main interface for passing features through the model.
    Provides type-safe access to feature groups with IDE autocomplete.

    Example:
        >>> features = AllAtomFeatures.from_featurizer(featurizer)
        >>> features.token.restype.shape  # [n_token, 32]
        >>> features.reference.ref_pos.shape  # [n_atom, 3]
        >>> features.to("cuda")
        >>> features.pad(n_token_pad=10, n_atom_pad=20)
    """

    token: TokenFeatures
    reference: ReferenceFeatures
    chain_perm: ChainPermFeatures
    extra: ExtraFeatures
    mask: MaskFeatures
    mpnn: MPNNFeatures

    def pad(self, n_max_tokens: int, n_max_atoms: int) -> "AllAtomFeatures":
        """Pad all features to n_max_tokens/n_max_atoms.

        Uses the FeatureType from each field's annotation to determine
        which dimensions to pad:
        - TOKEN: pad dim 0 by n_token_pad
        - TOKEN_PAIR: pad dims 0 and 1 by n_token_pad
        - ATOM: pad dim 0 by n_atom_pad
        - ATOM_PAIR: pad dims 0 and 1 by n_atom_pad
        """
        assert n_max_tokens >= self.num_tokens
        assert n_max_atoms >= self.num_atoms

        n_token_pad = n_max_tokens - self.num_tokens
        n_atom_pad = n_max_atoms - self.num_atoms

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
            if isinstance(fg, FeatureGroup):
                kwargs[f.name] = fg.pad(pad_fns)
            else:
                kwargs[f.name] = fg

        return type(self)(**kwargs)

    @property
    def num_tokens(self) -> int:
        return self.token.token_index.shape[0]

    @property
    def num_atoms(self) -> int:
        return self.reference.ref_pos.shape[0]
