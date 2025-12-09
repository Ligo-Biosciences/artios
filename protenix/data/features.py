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

import logging
from dataclasses import asdict, dataclass, fields
from enum import Enum, auto
from functools import partial
from typing import Annotated, Callable, TypeVar

import torch
from torch import Tensor

from protenix.model.utils import pad_tensor
from protenix.utils.typing import Bool, Float, Int, typecheck

logger = logging.getLogger(__name__)


# =============================================================================
# Feature metadata types
# =============================================================================


class FeatureRole(Enum):
    """Role of the feature in the model."""

    INPUT = auto()
    LABEL = auto()
    MASK = auto()
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
    description: str = ""


def Feature(
    jaxtype,
    *,
    type: FeatureType,
    encoding: EncodingType | None = None,
    role: FeatureRole = FeatureRole.INPUT,
    description: str = "",
):
    """Annotate a jaxtyped tensor with feature metadata.

    Usage:
        restype: Feature(
            Int[Tensor, "n_token 32"],
            encoding=EncodingType.ONE_HOT,
            type=FeatureType.TOKEN,
            description="Residue type one-hot encoding",
        )
    """
    return Annotated[jaxtype, FeatureMeta(encoding, type, description)]


# =============================================================================
# Base feature group class
# =============================================================================


T = TypeVar("T", bound="FeatureGroup")


@dataclass
class FeatureGroup:
    """Base class with common utilities for feature containers."""

    def to_dict(self) -> dict[str, Tensor]:
        """Convert to flat dictionary of tensors."""
        return {k: v for k, v in asdict(self).items() if torch.is_tensor(v)}

    def to(self: T, device: torch.device | str) -> T:
        """Move all tensors to specified device."""
        kwargs = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                kwargs[field.name] = value.to(device)
            else:
                kwargs[field.name] = value
        return type(self)(**kwargs)

    def pad(
        self: T,
        pad_fns: dict[FeatureType, Callable[[Tensor], Tensor]],
    ) -> T:
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


# =============================================================================
# Token-level features [n_token, ...]
# =============================================================================


@typecheck
@dataclass
class TokenFeatures(FeatureGroup):
    """Token-level features from AlphaFold3 SI Chapter 2.8.

    All tensors have shape [n_token, ...] where n_token is the number of tokens.
    """

    token_index: Feature(
        Int[Tensor, "n_token"],
        type=FeatureType.TOKEN,
        description="Token index in the sequence",
    )
    residue_index: Feature(
        Int[Tensor, "n_token"],
        type=FeatureType.TOKEN,
        description="Residue index in the chain",
    )
    asym_id: Feature(
        Int[Tensor, "n_token"],
        type=FeatureType.TOKEN,
        description="Asymmetric unit ID",
    )
    entity_id: Feature(
        Int[Tensor, "n_token"],
        encoding=EncodingType.IDENTITY,
        type=FeatureType.TOKEN,
        description="Entity ID",
    )
    sym_id: Feature(
        Int[Tensor, "n_token"],
        encoding=EncodingType.IDENTITY,
        type=FeatureType.TOKEN,
        description="Symmetry ID",
    )
    restype: Feature(
        Int[Tensor, "n_token 32"],
        encoding=EncodingType.ONE_HOT,
        type=FeatureType.TOKEN,
        description="Residue type one-hot encoding",
    )

    token_bonds: Feature(
        Int[Tensor, "n_token n_token"],
        type=FeatureType.TOKEN_PAIR,
        encoding=EncodingType.IDENTITY,
        description="Indicator for ligand-ligand and non-polymer bonds",
    )

    @property
    def num_tokens(self) -> int:
        return self.token_index.shape[0]


# =============================================================================
# Atom-level features [n_atom, ...]
# =============================================================================


@typecheck
@dataclass
class ReferenceFeatures(FeatureGroup):
    """Reference conformer features from AlphaFold3 SI Chapter 2.8.

    All tensors have shape [n_atom, ...] where n_atom is the number of atoms.
    """

    ref_pos: Feature(Float[Tensor, "n_atom 3"], type=FeatureType.ATOM)
    ref_mask: Feature(
        Int[Tensor, "n_atom"], type=FeatureType.ATOM
    )  # TODO: should be boolean
    ref_element: Feature(
        Int[Tensor, "n_atom 128"], type=FeatureType.ATOM
    )  # one-hot encoded
    ref_charge: Feature(Int[Tensor, "n_atom"], type=FeatureType.ATOM)
    ref_atom_name_chars: Feature(
        Int[Tensor, "n_atom 4 64"], type=FeatureType.ATOM
    )  # character encoded
    ref_space_uid: Feature(Int[Tensor, "n_atom"], type=FeatureType.ATOM)

    @property
    def num_atoms(self) -> int:
        return self.ref_pos.shape[0]


@typecheck
@dataclass
class ChainPermFeatures(FeatureGroup):
    """Chain permutation features for multi-chain alignment.

    Used for permutation-invariant training with symmetric chains.
    All tensors have shape [n_atom].
    """

    mol_id: Int[Tensor, "n_atom"]
    mol_atom_index: Int[Tensor, "n_atom"]
    entity_mol_id: Int[Tensor, "n_atom"]


@typecheck
@dataclass
class ExtraFeatures(FeatureGroup):
    """Additional features not in AlphaFold3 SI Table 5.

    Contains atom-to-token mappings and molecule type indicators.
    """

    # Atom-level [n_atom]
    atom_to_token_idx: Int[Tensor, "n_atom"]
    atom_to_tokatom_idx: Int[Tensor, "n_atom"]
    is_protein: Int[Tensor, "n_atom"]
    is_ligand: Int[Tensor, "n_atom"]
    is_dna: Int[Tensor, "n_atom"]
    is_rna: Int[Tensor, "n_atom"]

    # Token-level [n_token]
    is_protein_token: Bool[Tensor, "n_token"]
    is_ligand_token: Bool[Tensor, "n_token"]
    is_dna_token: Bool[Tensor, "n_token"]
    is_rna_token: Bool[Tensor, "n_token"]

    # Global
    resolution: Float[Tensor, "1"]


# =============================================================================
# Mask features
# =============================================================================


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

    # Token-level [n_token]
    token_mask: Bool[Tensor, "n_token"]


# =============================================================================
# Target features (for conditional generation)
# =============================================================================


@typecheck
@dataclass
class TargetFeatures(FeatureGroup):
    """Target-specific features for conditional generation.

    Used when generating structures conditioned on a target molecule.
    All tensors have shape [n_token, ...].
    """

    target_centre_coordinate: Float[Tensor, "n_token 3"]
    target_restype: Float[Tensor, "n_token 32"]
    is_target_token: Int[Tensor, "n_token"]
    is_hotspot: Int[Tensor, "n_token"]


# =============================================================================
# MPNN features (for sequence design)
# =============================================================================


@typecheck
@dataclass
class MPNNFeatures(FeatureGroup):
    """Features for MPNN-style sequence design.

    Contains backbone coordinates and masks for protein design.
    """

    mpnn_centre_coordinate: Float[Tensor, "n_token 3"]
    mpnn_centre_atom_mask: Float[Tensor, "n_token"]
    mpnn_backbone_coords: Float[Tensor, "n_token 5 3"]  # N, CA, C, O, CB
    mpnn_backbone_mask: Float[Tensor, "n_token 5"]


# =============================================================================
# Motif features
# =============================================================================


@typecheck
@dataclass
class MotifFeatures(FeatureGroup):
    """Motif features for motif scaffolding."""

    # Atom-level [n_atom]
    is_motif: Bool[Tensor, "n_atom"]
    is_motif_related: Bool[Tensor, "n_atom"]

    # Token-level [n_token]
    is_motif_token: Int[Tensor, "n_token"]
    is_motif_related_token: Int[Tensor, "n_token"]
    motif_restype: Float[Tensor, "n_token 32"]
    motif_coordinate: Float[Tensor, "n_token 3"]


# =============================================================================
# Labels (supervision signals)
# =============================================================================


@typecheck
@dataclass
class StructureLabels(FeatureGroup):
    """Ground truth labels for structure prediction training.

    Contains coordinates and masks for computing losses.
    """

    # Atom-level [n_atom]
    coordinate: Float[Tensor, "n_atom 3"]
    coordinate_mask: Bool[Tensor, "n_atom"]
    atom_loss_mask: Bool[Tensor, "n_atom"]

    # Token-level [n_token]
    centre_coordinate: Float[Tensor, "n_token 3"]
    has_centre_atom: Bool[Tensor, "n_token"]
    token_loss_mask: Bool[Tensor, "n_token"]
    dense_to_flat_atom_mask: Bool[Tensor, "n_token n_dense_atoms"]


# =============================================================================
# Main container
# =============================================================================


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
    target: TargetFeatures
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
