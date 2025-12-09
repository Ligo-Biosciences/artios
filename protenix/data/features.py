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
    # role: FeatureRole = FeatureRole.INPUT
    description: str = ""


def Feature(  # TODO: make this into a nice class instead of function
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
    return Annotated[jaxtype, FeatureMeta(encoding, type, role, description)]


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
    ref_space_uid: Feature(Int[Tensor, "n_atom"])


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

    def __post_init__(self):
        """Validate label consistency."""
        if torch.isnan(self.coordinate).any():
            logger.error("NaN values found in coordinate labels")

        # Check mask consistency
        if not torch.all(self.coordinate_mask <= self.atom_loss_mask + 1):
            logger.warning("coordinate_mask and atom_loss_mask may be inconsistent")


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
    """

    token: TokenFeatures
    reference: ReferenceFeatures
    chain_perm: ChainPermFeatures
    extra: ExtraFeatures
    mask: MaskFeatures
    target: TargetFeatures
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
            target=self.target.to(device),
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
        result.update(self.target.to_dict())
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
            target=TargetFeatures.from_dict(d),
            mpnn=MPNNFeatures.from_dict(d),
        )
