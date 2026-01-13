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
            # Convert string cif_file to CIFFile object format if needed
            if "cif_file" in spec_with_name:
                cif_file_value = spec_with_name["cif_file"]
                if isinstance(cif_file_value, str):
                    spec_with_name["cif_file"] = {"path": cif_file_value}

            return SequenceDesignInput(
                **spec_with_name
            )  # ty: ignore[invalid-argument-type]

        case TaskType.PACKING:
            # Conveert string cif_file to CIFFile object format if needed
            if "cif_file" in spec_with_name:
                cif_file_value = spec_with_name["cif_file"]
                if isinstance(cif_file_value, str):
                    spec_with_name["cif_file"] = {"path": cif_file_value}

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
    """Maps a residue in the motif CIF to its full ligand structure.

    Example:
        {
            "cif_res_name": "ZN",
            "full_ligand": {"type": "ccd", "codes": ["ZN"]}
        }
    """

    cif_res_name: str = Field(
        ..., description="Residue name as it appears in motif CIF"
    )
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

    Examples:
        {"chain_id": "A", "res_id": 94, "residue": "HIS"}
        {"chain_id": "A", "res_id": 94, "residue": "H"}  # 1-letter also accepted
    """

    chain_id: str = Field(..., description="Chain identifier")
    res_id: int = Field(..., ge=1, description="Residue number (1-based)")
    residue: str = Field(..., description="Amino acid (1-letter or 3-letter code)")

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

    Given a poly-glycine backbone structure with ligand(s), predict the amino acid
    sequence that would fold into that structure and bind the ligand.

    The CIF file should contain:
    - Poly-glycine protein chain(s) with backbone atoms (N, CA, C, O)
    - Ligand(s) with full atomic coordinates

    Example:
        {
            "name": "carbonic_anhydrase_redesign",
            "cif_file": "polyG_backbone.cif",
            "fixed_residues": [
                {"chain_id": "A", "res_id": 94, "residue": "HIS"},
                {"chain_id": "A", "res_id": 96, "residue": "HIS"},
                {"chain_id": "A", "res_id": 119, "residue": "H"}
            ],
            "ligands": [
                {"type": "smiles", "smiles": "CC(C)c1ccc([S:1](=[O:2])(=[O:3])[N:4])cc1"}
            ]
        }
    """

    name: str | None = Field(default=None, description="Job identifier")
    cif_file: CIFFile = Field(
        ..., description="Poly-glycine backbone CIF with ligand(s)"
    )
    fixed_residues: list[FixedResidue] = Field(
        default_factory=list,
        description="Residues with constrained amino acid identity",
    )
    ligands: list[FullLigand] = Field(
        ..., description="Ligands present in the structure."
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


# ---------------------------------------------------------------------------
# Packing models
# ---------------------------------------------------------------------------


class PackingInput(BaseModel):
    name: str | None = Field(default=None, description="Job identifier")
    cif_file: CIFFile = Field(
        ..., description="Poly-glycine backbone CIF with ligand(s)"
    )
    fasta_file: FastaFile = Field(..., description="Fasta file with sequence")
    ligands: list[FullLigand] = Field(..., description="Ligands to pack.")


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
        resolve_paths: If True, resolve cif_file path relative to the JSON file's directory

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

    # Convert cif_file string to CIFFile object format and resolve path
    if "cif_file" in data:
        cif_file_value = data["cif_file"]
        # Accept either string or dict with "path" key
        if isinstance(cif_file_value, str):
            cif_path = Path(cif_file_value)
        else:
            cif_path = Path(cif_file_value["path"])

        if resolve_paths and not cif_path.is_absolute():
            cif_path = file_path.parent / cif_path

        data["cif_file"] = {"path": str(cif_path)}

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
