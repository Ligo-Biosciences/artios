"""Enumerate every atom name in a motif's residues + ligands.

For protein residues, atom names are canonical PDB names sourced
from `RESIDUE_ATOMS`. For ligand residues, atom names are produced
by the same `build_ligand` path the production pipeline runs at
launch — so the names returned here are exactly what every
downstream consumer (folded structures, distance-metric lookup,
etc.) sees.

The eval-bot's distance-JSON workflow uses this to surface "what
atoms exist in this motif" to the agent / operator, since for
ligand atoms the names aren't guessable a priori — they fall out
of `smiles_to_atom_info` + `rename_atoms_to_match_cif_names`.

Fail-loud on motif inconsistencies. If a CIF has an atom name that
doesn't appear in the build-pipeline output (e.g. a non-PDB name
on a protein residue, or a ligand atom name that the JSON's
atom_map doesn't account for), we raise rather than silently
producing a phantom record. Those mismatches are upstream-bug
class — surfacing them at enumeration time is cheaper than
chasing them through training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from artios.common.atom_types import RESIDUE_ATOMS
from artios.data.array_builder import build_ligand
from artios.data.parsing.json import BackboneInput, SMILESLigand
from artios.data.parsing.motif import Motif, MotifLigand, MotifResidue


@dataclass(frozen=True)
class AtomRecord:
    """One atom in the motif, identified by canonical fields.

    The four canonical fields are exactly what `DistanceAtomSpec`
    requires, so a record can be pasted straight into a distance
    JSON pair. `component_kind` is metadata for downstream rendering
    dispatch.
    """

    canonical_chain_id: str
    canonical_res_id: int
    res_name: str
    atom_name: str
    component_kind: Literal["protein", "ligand"]


@dataclass(frozen=True)
class MotifComponent:
    """One residue/ligand in the motif, with everything the renderer
    needs to draw it.

    For protein components: `atom_names` is the full PDB-name set
    from `RESIDUE_ATOMS[res_name]`.

    For ligand components: `atom_names` is the post-rename atom-name
    array from `build_ligand` (motif atoms carry their CIF names;
    non-motif atoms carry sequential element-counter names like
    `C5`, `N7`). `smiles` is the labelled SMILES (with `[X:N]`
    notation), and `atom_map` is the CIF-name → integer-N dict.
    """

    canonical_chain_id: str
    canonical_res_id: int
    res_name: str
    component_kind: Literal["protein", "ligand"]
    atom_names: tuple[str, ...]
    smiles: str | None = None
    atom_map: dict[str, int] | None = None


def build_motif_components(
    backbone_input: BackboneInput,
) -> list[MotifComponent]:
    """Build the motif components (protein + ligand) for a parsed
    backbone input. Reused by `enumerate_motif_atom_records` and the
    eval-bot rendering tools.
    """
    if backbone_input.motif is None:
        return []

    motif = Motif.from_spec(backbone_input.motif)
    components: list[MotifComponent] = []

    for residue in motif.residues:
        components.append(_protein_component(residue))

    for ligand in motif.ligands:
        components.append(_ligand_component(ligand))

    return components


def enumerate_motif_atom_records(
    backbone_input: BackboneInput,
) -> list[AtomRecord]:
    """Flat list of every atom in the motif, ordered by component
    declaration and within-component canonical order. Drop-in for
    operator/agent reasoning.
    """
    components = build_motif_components(backbone_input)
    return [
        AtomRecord(
            canonical_chain_id=c.canonical_chain_id,
            canonical_res_id=c.canonical_res_id,
            res_name=c.res_name,
            atom_name=name,
            component_kind=c.component_kind,
        )
        for c in components
        for name in c.atom_names
    ]


def _protein_component(residue: MotifResidue) -> MotifComponent:
    """Build a protein MotifComponent. Uses the canonical
    `RESIDUE_ATOMS` set as the atom list; verifies that every CIF
    atom name on the residue appears there."""
    canonical = RESIDUE_ATOMS.get(residue.res_name)
    if canonical is None:
        raise ValueError(
            f"motif inconsistency: residue {residue.res_name} "
            f"(chain {residue.chain_id}, res_id {residue.res_id}) "
            f"is not in RESIDUE_ATOMS — non-standard amino acids "
            f"are not supported in the distance-JSON workflow."
        )
    canonical_set = set(canonical)
    for cif_name in residue.atom_names:
        if cif_name not in canonical_set:
            raise ValueError(
                f"motif inconsistency: motif CIF has atom name "
                f"{cif_name!r} on residue {residue.res_name} "
                f"(chain {residue.chain_id}, res_id {residue.res_id}), "
                f"which is not a PDB-canonical name for that residue. "
                f"Expected one of: {sorted(canonical_set)}."
            )
    return MotifComponent(
        canonical_chain_id=residue.chain_id,
        canonical_res_id=residue.res_id,
        res_name=residue.res_name,
        component_kind="protein",
        atom_names=tuple(canonical),
    )


def _ligand_component(ligand: MotifLigand) -> MotifComponent:
    """Build a ligand MotifComponent. Runs `build_ligand` (the same
    function `array_builder.build_motif_ligand` calls) so the atom
    names match production exactly. Carries the labelled SMILES +
    atom_map for downstream rendering."""
    atom_info = build_ligand(ligand.full_ligand)
    full_atom_names = tuple(str(n) for n in atom_info.atom_array.atom_name)

    full_atom_set = set(full_atom_names)
    for cif_name in ligand.atom_names:
        if cif_name not in full_atom_set:
            raise ValueError(
                f"motif inconsistency: motif CIF ligand atom "
                f"{cif_name!r} (chain {ligand.chain_id}, res_id "
                f"{ligand.res_id}, cif_res_name {ligand.cif_res_name}) "
                f"is not present in the build_ligand output. The "
                f"JSON's atom_map and the CIF's atom names disagree. "
                f"Built atom names: {sorted(full_atom_set)}."
            )

    smiles: str | None = None
    atom_map: dict[str, int] | None = None
    if isinstance(ligand.full_ligand, SMILESLigand):
        smiles = ligand.full_ligand.smiles
        atom_map = ligand.full_ligand.atom_map

    return MotifComponent(
        canonical_chain_id=ligand.chain_id,
        canonical_res_id=ligand.res_id,
        res_name=ligand.cif_res_name,
        component_kind="ligand",
        atom_names=full_atom_names,
        smiles=smiles,
        atom_map=atom_map,
    )
