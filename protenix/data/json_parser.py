"""Pydantic-based JSON parser for inference inputs.

Supports:
- Motif scaffolding: Design a protein around a structural motif
- Sequence design: Predict sequence for a given backbone structure
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from protenix.common.residue_names import (
    PROTEIN_COMMON_ONE_TO_THREE,
    PROTEIN_COMMON_THREE_TO_ONE,
)
from protenix.data.ccd import is_valid_ccd_code
from protenix.data.rdkit import is_valid_smiles, parse_smiles

# ---------------------------------------------------------------------------
# Inference input models
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Supported inference task types."""

    BACKBONE = "backbone"
    SEQUENCE_DESIGN = "sequence_design"
    PACKING = "packing"


class InferenceInputDict(BaseModel):
    """Structured input format for inference pipeline.

    Each item in the dataset JSON list should follow this format.

    Example:
        {
            "name": "carbonic-motif",
            "task": "backbone",
            "spec": {
                "design": [{"length": 256}],
                "motif": {"cif_file": "motif.cif", "ligands": [...]}
            }
        }
    """

    name: str = Field(..., description="Job identifier")
    task: TaskType = Field(..., description="Type of inference task")
    spec: dict[str, Any] = Field(..., description="Task-specific input data")


def parse_input_dict(
    input_dict: dict[str, Any],
) -> BackboneInput | SequenceDesignInput | PackingInput:
    """Parse a single input dict from the inference dataset JSON.

    Args:
        input_dict: Dictionary containing 'name', 'task', and 'spec' keys

    Returns:
        Parsed and validated XInput object (BackboneInput or SequenceDesignInput or PackingInput)

    Raises:
        pydantic.ValidationError: If input_dict doesn't match expected structure
        ValueError: If task type is not supported
    """
    validated_input = InferenceInputDict(**input_dict)

    # Add name to spec
    spec_with_name = {"name": validated_input.name, **validated_input.spec}

    match validated_input.task:
        case TaskType.BACKBONE:
            # Convert string cif_file to CIFFile object format if needed
            motif_spec = spec_with_name.get("motif")
            if isinstance(motif_spec, dict) and "cif_file" in motif_spec:
                cif_file_value = motif_spec["cif_file"]
                if isinstance(cif_file_value, str):
                    motif_spec["cif_file"] = {"path": cif_file_value}

            return BackboneInput(**spec_with_name)  # ty: ignore[invalid-argument-type]

        case TaskType.SEQUENCE_DESIGN:
            # Convert string npz_file to NPZFile object format if needed
            if "npz_file" in spec_with_name:
                npz_file_value = spec_with_name["npz_file"]
                if isinstance(npz_file_value, str):
                    spec_with_name["npz_file"] = {"path": npz_file_value}

            return SequenceDesignInput(
                **spec_with_name
            )  # ty: ignore[invalid-argument-type]

        case TaskType.PACKING:
            # Convert string npz_file to NPZFile object format if needed
            if "npz_file" in spec_with_name:
                npz_file_value = spec_with_name["npz_file"]
                if isinstance(npz_file_value, str):
                    spec_with_name["npz_file"] = {"path": npz_file_value}

            # Convert string fasta_file to FastaFile object format if needed
            if "fasta_file" in spec_with_name:
                fasta_file_value = spec_with_name["fasta_file"]
                if isinstance(fasta_file_value, str):
                    spec_with_name["fasta_file"] = {"path": fasta_file_value}

            return PackingInput(**spec_with_name)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# Common models
# ---------------------------------------------------------------------------


class CIFFile(BaseModel):
    """A CIF/mmCIF structure file.

    Used by both motif scaffolding and sequence design inputs.

    Example:
        {"path": "path/to/structure.cif"}
    """

    path: Path = Field(..., description="Path to CIF file")

    @field_validator("path")
    @classmethod
    def validate_file_suffix(cls, v: Path) -> Path:
        valid_suffixes = {".cif", ".mmcif"}
        if v.suffix.lower() not in valid_suffixes:
            raise ValueError(f"File must be .cif or .mmcif, got '{v.suffix}'")
        return v


class FastaFile(BaseModel):
    """A FASTA file.

    Used by packing inputs.

    Example:
        {"path": "path/to/sequence.fasta"}
    """

    path: Path = Field(..., description="Path to FASTA file")

    @field_validator("path")
    @classmethod
    def validate_file_suffix(cls, v: Path) -> Path:
        valid_suffixes = {".fasta", ".fa"}
        if v.suffix.lower() not in valid_suffixes:
            raise ValueError(f"File must be .fasta or .fa, got '{v.suffix}'")
        return v


class NPZFile(BaseModel):
    """An NPZ file containing a saved AtomArray.

    Used by sequence design inputs to load the atom array from backbone inference.

    Example:
        {"path": "path/to/backbone_output.npz"}
    """

    path: Path = Field(..., description="Path to NPZ file")

    @field_validator("path")
    @classmethod
    def validate_file_suffix(cls, v: Path) -> Path:
        if v.suffix.lower() != ".npz":
            raise ValueError(f"File must be .npz, got '{v.suffix}'")
        return v


# ---------------------------------------------------------------------------
# Motif scaffolding models
# ---------------------------------------------------------------------------


class Backbone(BaseModel):
    """A protein chain to be designed."""

    length: int = Field(..., description="Number of residues")

    @field_validator("length")
    @classmethod
    def validate_length(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Protein length must be at least 1, got {v}")
        if v > 2000:
            raise ValueError(f"Protein length must be at most 2000, got {v}")
        return v


class CCDLigand(BaseModel):
    """Ligand from Chemical Component Dictionary.

    Examples:
        {"type": "ccd", "codes": ["ZN"]}
        {"type": "ccd", "codes": ["HEM", "FE"]}
    """

    type: Literal["ccd"] = "ccd"
    codes: list[str] = Field(..., min_length=1)

    @field_validator("codes")
    @classmethod
    def validate_codes(cls, v: list[str]) -> list[str]:
        for code in v:
            if not code.isupper() or not code.isalnum():
                raise ValueError(
                    f"CCD code must be uppercase alphanumeric, got '{code}'"
                )
            if not is_valid_ccd_code(code):
                raise ValueError(
                    f"CCD code '{code}' not found in Chemical Component Dictionary"
                )
        return v


class SMILESLigand(BaseModel):
    """Ligand from SMILES string.

    Examples:
        {"type": "smiles", "smiles": "Cc1ccc(S(=O)(=O)N)cc1"}
        {"type": "smiles", "smiles": "[S:1](=[O:2])N", "atom_map": {"S1": 1, "O1": 2}}
    """

    type: Literal["smiles"] = "smiles"
    smiles: str = Field(..., min_length=1)
    atom_map: dict[str, int] | None = Field(
        default=None,
        description="Maps CIF atom names to SMILES atom map numbers",
    )

    @field_validator("smiles")
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        if not is_valid_smiles(v):
            raise ValueError(f"Invalid SMILES string: '{v}'")
        return v

    @model_validator(mode="after")
    def validate_atom_map_matches_smiles(self) -> SMILESLigand:
        """Validate that atom_map numbers exist in the SMILES string."""
        if self.atom_map is None:
            return self

        mol = parse_smiles(self.smiles)
        if mol is None:
            # Already caught by field validator, but be safe
            return self

        # Get all atom map numbers from the SMILES
        smiles_map_nums = {
            atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() != 0
        }

        # Check that all atom_map values exist in SMILES
        for cif_name, map_num in self.atom_map.items():
            if map_num not in smiles_map_nums:
                raise ValueError(
                    f"Atom map number {map_num} (for CIF atom '{cif_name}') not found "
                    f"in SMILES. Use [atom:{map_num}] notation in the SMILES string. "
                    f"Available map numbers in SMILES: {sorted(smiles_map_nums)}"
                )

        return self


FullLigand = Annotated[
    Union[CCDLigand, SMILESLigand],
    Field(discriminator="type"),
]


class LigandMapping(BaseModel):
    """Maps a ligand in the CIF to its full ligand structure.

    The chain_id is required to distinguish between multiple instances of the
    same ligand type (e.g., two ZN ions with different chain_ids).

    Example:
        {
            "cif_res_name": "ZN",
            "chain_id": "3",
            "full_ligand": {"type": "ccd", "codes": ["ZN"]}
        }
    """

    cif_res_name: str = Field(..., description="Residue name as it appears in the CIF")
    chain_id: str = Field(..., description="Chain ID as it appears in the CIF")
    full_ligand: FullLigand


class MotifSpec(BaseModel):
    """Motif specification: CIF file with partial coordinates + ligand mappings.

    The CIF file contains:
    - Partial protein sidechains (auto-detected via hetero=False, expanded from CCD)
    - Ligand fragments (require explicit mapping in `ligands` list)

    Example:
        {
            "cif_file": "motifs/active_site.cif",
            "ligands": [
                {"cif_res_name": "ZN", "full_ligand": {"type": "ccd", "codes": ["ZN"]}}
            ]
        }
    """

    cif_file: CIFFile = Field(..., description="Motif CIF file")
    ligands: list[LigandMapping] = Field(
        default_factory=list,
        description="Ligand mappings (protein residues are auto-detected)",
    )


class BackboneInput(BaseModel):
    """Top-level input for backbone inference with motif scaffolding capabilities.

    Example:
        {
            "name": "my_design",
            "design": [{"length": 260}],
            "motif": {
                "cif_file": "motif.cif",
                "ligands": [...]
            }
        }
    """

    name: str | None = Field(default=None, description="Job identifier")
    design: list[Backbone] = Field(
        ..., min_length=1, description="Protein chains to design"
    )
    motif: MotifSpec | None = Field(default=None, description="Motif specification")


# ---------------------------------------------------------------------------
# Sequence design models
# ---------------------------------------------------------------------------


class FixedResidue(BaseModel):
    """A residue with a fixed (constrained) amino acid identity.

    Used to specify catalytic residues, binding site residues, or other
    positions where the amino acid identity is known/required.

    There are two types of fixed residues:
    - Identity-fixed only (is_motif_residue=False): The residue type is constrained,
      but sidechain coordinates are not specified. Example: requiring an ARG near
      an active site without specific coordinate preferences.
    - Motif residues (is_motif_residue=True): The residue has full sidechain
      coordinates from motif scaffolding output. These residues will have their
      sidechains preserved for MPNN attention.

    Examples:
        {"chain_id": "A", "res_id": 94, "residue": "HIS", "is_motif_residue": true}
        {"chain_id": "A", "res_id": 200, "residue": "ARG"}  # identity-fixed only
    """

    chain_id: str = Field(..., description="Chain identifier")
    res_id: int = Field(..., ge=1, description="Residue number (1-based)")
    residue: str = Field(..., description="Amino acid (1-letter or 3-letter code)")
    is_motif_residue: bool = Field(
        default=False,
        description="True if this residue has sidechain coordinates from motif scaffolding",
    )

    @field_validator("residue")
    @classmethod
    def validate_and_normalize_residue(cls, v: str) -> str:
        """Validate and normalize residue to 3-letter code."""
        code_upper = v.upper()

        if code_upper in PROTEIN_COMMON_THREE_TO_ONE:
            return code_upper
        elif code_upper in PROTEIN_COMMON_ONE_TO_THREE:
            return PROTEIN_COMMON_ONE_TO_THREE[code_upper]
        else:
            raise ValueError(
                f"Invalid residue code '{v}'. "
                f"Must be a standard amino acid (1-letter: {sorted(PROTEIN_COMMON_ONE_TO_THREE.keys())}, "
                f"3-letter: {sorted(PROTEIN_COMMON_THREE_TO_ONE.keys())})"
            )


class SequenceDesignInput(BaseModel):
    """Top-level input for sequence design inference.

    Given a backbone structure from motif scaffolding, predict the amino acid
    sequence that would fold into that structure and bind any ligands.

    The NPZ file is the saved AtomArray from backbone inference (before stitching).
    It contains all necessary information including:
    - Protein backbone coordinates
    - Ligand coordinates and annotations
    - Motif residue annotations (is_motif, is_motif_related)
    - Chain IDs for all entities

    Fixed residues can optionally be provided for additional constraints:
    - Motif residues (is_motif_residue=True): Have sidechain coordinates from motif
      scaffolding. Their sidechains are preserved for MPNN attention.
    - Identity-fixed only (is_motif_residue=False): Residue type is constrained but
      no coordinate preferences. Used for additional constraints like "put an ARG here".

    Note: Protein chain must always be "0". This is enforced by validation.

    Example:
        {
            "name": "carbonic_anhydrase_redesign",
            "npz_file": "backbone_output.npz",
            "fixed_residues": [
                {"chain_id": "0", "res_id": 94, "residue": "HIS", "is_motif_residue": true},
                {"chain_id": "0", "res_id": 96, "residue": "HIS", "is_motif_residue": true},
                {"chain_id": "0", "res_id": 119, "residue": "HIS", "is_motif_residue": true}
            ]
        }
    """

    name: str = Field(..., description="Job identifier")
    npz_file: NPZFile = Field(
        ...,
        description="NPZ file from backbone inference containing the full AtomArray",
    )
    fixed_residues: list[FixedResidue] = Field(
        default_factory=list,
        description="Optional additional residue constraints. Motif residues (is_motif_residue=True) have sidechain coordinates.",
    )

    @model_validator(mode="after")
    def validate_no_duplicate_fixed_residues(self) -> SequenceDesignInput:
        """Ensure no residue is fixed twice."""
        seen: set[tuple[str, int]] = set()
        for fr in self.fixed_residues:
            key = (fr.chain_id, fr.res_id)
            if key in seen:
                raise ValueError(
                    f"Duplicate fixed residue: chain '{fr.chain_id}', res_id {fr.res_id}"
                )
            seen.add(key)
        return self

    @model_validator(mode="after")
    def validate_protein_chain_id(self) -> SequenceDesignInput:
        """Ensure all fixed residues are on protein chain '0'."""
        for fr in self.fixed_residues:
            if fr.chain_id != "0":
                raise ValueError(
                    f"Fixed residue chain_id must be '0' (protein chain), "
                    f"got '{fr.chain_id}' for res_id {fr.res_id}"
                )
        return self


# ---------------------------------------------------------------------------
# Packing models
# ---------------------------------------------------------------------------


class PackingInput(BaseModel):
    """Top-level input for sidechain packing inference.

    Given a backbone structure (from motif scaffolding) and designed sequences
    (from MPNN), predict sidechain conformations.

    The NPZ file contains an AtomArrayStack saved with dump_atomarray_stack().
    It includes:
    - Protein backbone coordinates (poly-glycine from motif scaffolding)
    - Ligand coordinates and annotations
    - All necessary chain/entity annotations

    The FASTA file contains the designed sequences from MPNN, one per model
    in the AtomArrayStack. The number of sequences must match the stack depth.

    Example:
        {
            "name": "pack_design",
            "npz_file": "backbone_stack.npz",
            "fasta_file": "designed_sequences.fasta"
        }
    """

    name: str | None = Field(default=None, description="Job identifier")
    npz_file: NPZFile = Field(
        ...,
        description="NPZ file containing AtomArrayStack from motif scaffolding",
    )
    fasta_file: FastaFile = Field(
        ..., description="FASTA file with designed sequences (one per model in stack)"
    )


# ---------------------------------------------------------------------------
# Legacy
# ---------------------------------------------------------------------------


def parse_sequence_design_json(
    file_path: str | Path,
    resolve_paths: bool = True,
) -> SequenceDesignInput:
    """Parse a sequence design input JSON file.

    Args:
        file_path: Path to the JSON file
        resolve_paths: If True, resolve npz_file path relative to the JSON file's directory

    Returns:
        Validated SequenceDesignInput object

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        pydantic.ValidationError: If the JSON doesn't match the schema
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    # Convert npz_file string to NPZFile object format and resolve path
    if "npz_file" in data:
        npz_file_value = data["npz_file"]
        # Accept either string or dict with "path" key
        if isinstance(npz_file_value, str):
            npz_path = Path(npz_file_value)
        else:
            npz_path = Path(npz_file_value["path"])

        if resolve_paths and not npz_path.is_absolute():
            npz_path = file_path.parent / npz_path

        data["npz_file"] = {"path": str(npz_path)}

    return SequenceDesignInput(**data)


def parse_motif_json(
    file_path: str | Path, resolve_paths: bool = True
) -> BackboneInput:
    """Parse a motif scaffolding input JSON file.

    Args:
        file_path: Path to the JSON file
        resolve_paths: If True, resolve cif_file path relative to the JSON file's directory

    Returns:
        Validated MotifInput object

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        pydantic.ValidationError: If the JSON doesn't match the schema
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    # Convert motif.cif_file string to CIFFile object format and resolve path
    if "motif" in data and "cif_file" in data["motif"]:
        cif_file_value = data["motif"]["cif_file"]
        # Accept either string or dict with "path" key
        if isinstance(cif_file_value, str):
            cif_path = Path(cif_file_value)
        else:
            cif_path = Path(cif_file_value["path"])

        if resolve_paths and not cif_path.is_absolute():
            cif_path = file_path.parent / cif_path

        data["motif"]["cif_file"] = {"path": str(cif_path)}

    return BackboneInput(**data)


def parse_packing_json(
    file_path: str | Path,
    resolve_paths: bool = True,
) -> PackingInput:
    """Parse a packing input JSON file.

    Args:
        file_path: Path to the JSON file
        resolve_paths: If True, resolve npz_file and fasta_file paths relative
            to the JSON file's directory

    Returns:
        Validated PackingInput object

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        pydantic.ValidationError: If the JSON doesn't match the schema
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    # Convert npz_file string to NPZFile object format and resolve path
    if "npz_file" in data:
        npz_file_value = data["npz_file"]
        if isinstance(npz_file_value, str):
            npz_path = Path(npz_file_value)
        else:
            npz_path = Path(npz_file_value["path"])

        if resolve_paths and not npz_path.is_absolute():
            npz_path = file_path.parent / npz_path

        data["npz_file"] = {"path": str(npz_path)}

    # Convert fasta_file string to FastaFile object format and resolve path
    if "fasta_file" in data:
        fasta_file_value = data["fasta_file"]
        if isinstance(fasta_file_value, str):
            fasta_path = Path(fasta_file_value)
        else:
            fasta_path = Path(fasta_file_value["path"])

        if resolve_paths and not fasta_path.is_absolute():
            fasta_path = file_path.parent / fasta_path

        data["fasta_file"] = {"path": str(fasta_path)}

    return PackingInput(**data)
