"""Typed feature containers for all-atom structure modeling.

This module provides dataclass-based containers for features with runtime
shape and dtype validation via jaxtyping + beartype.

Example usage:
    >>> features = AllAtomFeatures.from_featurizer(featurizer)
    >>> features = features.to("cuda")
    >>> features.token.restype  # IDE autocomplete works
    >>> features.to_dict()  # backward compatibility

Feature metadata can be accessed via:
    >>> TokenFeatures.get_field_meta("restype")
    FeatureMeta(encoding=Encoding.ONE_HOT, ...)
"""

import logging
from dataclasses import asdict, dataclass, fields
from enum import Enum, auto
from functools import cached_property
from typing import Annotated, Any, TypeVar

import torch
from torch import Tensor

from protenix.utils.typing import Bool, Float, Int, typecheck

logger = logging.getLogger(__name__)


class FeatureRole(Enum):
    """Role of the feature in the model."""

    INPUT = auto()
    LABEL = auto()
    MASK = auto()  # masks can be input into model as well e.g. ref_mask
    AUXILIARY = auto()


class FeatureType(Enum):
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
        level: Granularity level (atom, token, pair, global)
        role: Role in the model (input, label, mask, auxiliary)
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


FeatureGroup = TypeVar("FeatureGroup", bound="FeatureGroup")


@dataclass
class FeatureGroup:
    """Base class with common utilities for feature containers."""

    def to_dict(self) -> dict[str, Tensor]:
        """Convert to flat dictionary of tensors."""
        return {k: v for k, v in asdict(self).items() if torch.is_tensor(v)}

    def pad(self, pad_fn: callable) -> FeatureGroup:
        """Pad all tensor fields in the feature group.

        Args:
            pad_fn (callable): Function that takes a tensor and returns a padded tensor

        Returns:
            FeatureGroup: New instance with padded tensors
        """
        padded_fields = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                padded_fields[field.name] = pad_fn(value)
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

    def __post_init__(self):
        """Validate label consistency."""
        if torch.isnan(self.coordinate).any():
            logger.error("NaN values found in coordinate labels")

        # Check mask consistency
        if not torch.all(self.coordinate_mask <= self.atom_loss_mask + 1):
            logger.warning("coordinate_mask and atom_loss_mask may be inconsistent")


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
    """

    token: TokenFeatures
    reference: ReferenceFeatures
    chain_perm: ChainPermFeatures
    extra: ExtraFeatures
    mask: MaskFeatures
    mpnn: MPNNFeatures

    def __post_init__(self):
        """Validate cross-feature consistency."""
        n_tokens = self.token.num_tokens
        n_atoms = self.reference.num_atoms

        # Validate token counts match
        if self.mask.token_mask.shape[0] != n_tokens:
            logger.error(
                f"Token count mismatch: token={n_tokens}, "
                f"mask={self.mask.token_mask.shape[0]}"
            )

        # Validate atom counts match
        if self.mask.atom_mask.shape[0] != n_atoms:
            logger.error(
                f"Atom count mismatch: reference={n_atoms}, "
                f"mask={self.mask.atom_mask.shape[0]}"
            )

        # Validate atom_to_token_idx is in valid range
        max_token_idx = self.extra.atom_to_token_idx.max()
        if max_token_idx >= n_tokens:
            logger.error(
                f"atom_to_token_idx contains invalid index {max_token_idx} "
                f"for {n_tokens} tokens"
            )

    @cached_property
    def num_tokens(self) -> int:
        return self.token.num_tokens

    @cached_property
    def num_atoms(self) -> int:
        return self.reference.num_atoms

    @property
    def device(self) -> torch.device:
        return self.token.token_index.device

    def to(self, device: torch.device | str) -> "AllAtomFeatures":
        """Move all features to specified device."""
        return AllAtomFeatures(
            token=self.token.to(device),
            reference=self.reference.to(device),
            chain_perm=self.chain_perm.to(device),
            extra=self.extra.to(device),
            mask=self.mask.to(device),
            mpnn=self.mpnn.to(device),
        )

    def to_dict(self) -> dict[str, Tensor]:
        """Flatten all features to single dictionary.

        Useful for backward compatibility with dict-based APIs.
        """
        result = {}
        result.update(self.token.to_dict())
        result.update(self.reference.to_dict())
        result.update(self.chain_perm.to_dict())
        result.update(self.extra.to_dict())
        result.update(self.mask.to_dict())
        result.update(self.mpnn.to_dict())
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AllAtomFeatures":
        """Construct from flat dictionary.

        Useful for loading from checkpoints or migrating from dict-based code.
        """
        return cls(
            token=TokenFeatures.from_dict(d),
            reference=ReferenceFeatures.from_dict(d),
            chain_perm=ChainPermFeatures.from_dict(d),
            extra=ExtraFeatures.from_dict(d),
            mask=MaskFeatures.from_dict(d),
            mpnn=MPNNFeatures.from_dict(d),
        )
