"""Load PDB/mmCIF structures as ArtiosPotts design targets.

A ``DesignTarget`` is a structure prepared the way ArtiosPotts saw proteins in
training: waters, hydrogens, crystallisation aids and glycans removed, protein
side chains stripped to the backbone, no motif. Residues are addressed by
author chain and residue number (``"A42"``, ``"A52B"`` with an insertion code),
as in PyMOL or LigandMPNN.

    target = load_target("complex.pdb")
    context = target.build_context(design_chains=["B"], fixed=parse_selection("B10-12"))
"""

from __future__ import annotations

import dataclasses
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import biotite.structure as struc
import numpy as np
import torch
from biotite.structure.io import pdb, pdbx

from artios.common import ccd_sets
from artios.configs.configs_data import data_configs
from artios.data import filter as filter_module
from artios.data.collate.collate import CollateContexts
from artios.data.features import FullContext
from artios.data.features.context import FixedResidueContext
from artios.data.features.featurizer import ContextGenerator
from artios.data.parsing import residue_repair
from artios.data.parsing.mmcif import MmcifReader
from artios.data.structure import AnnotatedStructure
from artios.data.structure import annotation as struct_annotation
from artios.data.structure.annotation import MotifAnnotation
from artios.data.utils import int_to_letters

# A bare chain ID is letters only ("A42", "AB12"); other chain IDs need a colon
# ("1:42", "A1:42").
_SELECTOR_RE = re.compile(
    r"^(?:(?P<qchain>[^:\s]+):|(?P<chain>[A-Za-z]+))"
    r"(?:(?P<start>-?\d+)(?P<start_ins>[A-Za-z]?)"
    r"(?:-(?P<end>-?\d+)(?P<end_ins>[A-Za-z]?))?)?$"
)


@dataclass(frozen=True)
class Residue:
    """A protein residue in author numbering."""

    chain: str
    number: int
    ins_code: str = ""
    res_name: str = ""
    canonical_res_name: str = ""
    """Standard parent residue the model sees (e.g. MET for MSE, UNK if none)."""

    @property
    def label(self) -> str:
        return f"{self.chain}{self.number}{self.ins_code}"


@dataclass(frozen=True)
class ResidueSelector:
    """A whole chain (``"A"``), one residue (``"A42"``) or a range (``"A10-20"``).

    Ranges are inclusive and follow file order, so insertion-coded residues
    between the two ends are included. Chain IDs that are not purely letters
    are written with a colon: ``"1:42"``.
    """

    chain: str
    start: tuple[int, str] | None = None
    end: tuple[int, str] | None = None

    @classmethod
    def parse(cls, text: str) -> ResidueSelector:
        m = _SELECTOR_RE.match(text.strip())
        if m is None:
            raise ValueError(
                f"Cannot parse residue selection {text!r}; expected e.g. "
                "'A', 'A42', 'A52B', 'A10-20' or '1:42'"
            )
        chain = m["qchain"] or m["chain"]
        if m["start"] is None:
            return cls(chain=chain)
        start = (int(m["start"]), m["start_ins"].upper())
        end = (int(m["end"]), m["end_ins"].upper()) if m["end"] is not None else start
        return cls(chain=chain, start=start, end=end)


def parse_selection(selection: str | Iterable[str] | None) -> list[ResidueSelector]:
    """Parse ``"A12 A15-20, B"`` (or a list of such strings) into selectors."""
    if selection is None:
        return []
    items = [selection] if isinstance(selection, str) else list(selection)
    tokens = [t for item in items for t in re.split(r"[\s,]+", item) if t]
    return [ResidueSelector.parse(t) for t in tokens]


@dataclass
class DesignTarget:
    """A structure prepared for ArtiosPotts, with per-token residue metadata."""

    name: str
    structure: AnnotatedStructure
    residues: tuple[Residue | None, ...]
    """One entry per token; ``None`` for non-protein tokens (ligands, ions)."""

    @property
    def protein_chains(self) -> list[str]:
        return list(dict.fromkeys(r.chain for r in self.residues if r is not None))

    @property
    def is_protein(self) -> np.ndarray:
        return np.array([r is not None for r in self.residues])

    def chain_mask(self, chains: Iterable[str]) -> np.ndarray:
        chains = set(chains)
        unknown = chains - set(self.protein_chains)
        if unknown:
            raise ValueError(
                f"Chains {sorted(unknown)} are not protein chains of {self.name}; "
                f"protein chains are {self.protein_chains}"
            )
        return np.array([r is not None and r.chain in chains for r in self.residues])

    def select(self, selectors: Sequence[ResidueSelector]) -> np.ndarray:
        """Token mask for ``selectors``. Raises if a selector matches nothing."""
        mask = np.zeros(len(self.residues), dtype=bool)
        for sel in selectors:
            hit = self._select_one(sel)
            if not hit.any():
                raise ValueError(f"Selection {sel} matches no residue in {self.name}")
            mask |= hit
        return mask

    def _select_one(self, sel: ResidueSelector) -> np.ndarray:
        in_chain = np.array(
            [r is not None and r.chain == sel.chain for r in self.residues]
        )
        if sel.start is None:
            return in_chain
        keys = [
            (r.number, r.ins_code) if r is not None else None for r in self.residues
        ]
        idx = np.flatnonzero(in_chain)
        positions = {keys[i]: i for i in idx}
        if sel.start not in positions or sel.end not in positions:
            missing = sel.start if sel.start not in positions else sel.end
            raise ValueError(
                f"Residue {sel.chain}{missing[0]}{missing[1]} not found in {self.name}"
            )
        lo, hi = positions[sel.start], positions[sel.end]
        if lo > hi:
            raise ValueError(f"Empty residue range {sel} in {self.name}")
        mask = np.zeros(len(self.residues), dtype=bool)
        mask[idx[(idx >= lo) & (idx <= hi)]] = True
        return mask

    def build_context(
        self,
        *,
        design_chains: Iterable[str] | None = None,
        fixed: Sequence[ResidueSelector] = (),
        redesign: Sequence[ResidueSelector] | None = None,
    ) -> FullContext:
        """Featurize the target with a design specification.

        Args:
            design_chains: Protein chains to design. Default: all of them.
                Residues of other protein chains keep their native identity.
            fixed: Residues within the design chains that keep their native
                identity.
            redesign: If given, only these residues are designed (``fixed``
                still applies on top).
        """
        mutable = self.mutable_mask(
            design_chains=design_chains, fixed=fixed, redesign=redesign
        )
        context = ContextGenerator(self.structure, pdb_id=self.name).get_full_context()
        keep = torch.from_numpy(self.is_protein & ~mutable)
        return dataclasses.replace(
            context,
            fixed_residues=FixedResidueContext(
                fixed_mask=keep,
                fixed_restype=context.token.restype.float() * keep[:, None],
            ),
        )

    def mutable_mask(
        self,
        *,
        design_chains: Iterable[str] | None = None,
        fixed: Sequence[ResidueSelector] = (),
        redesign: Sequence[ResidueSelector] | None = None,
    ) -> np.ndarray:
        if design_chains is None:
            mutable = self.is_protein.copy()
        else:
            mutable = self.chain_mask(design_chains)
        if redesign is not None:
            mutable &= self.select(redesign)
        if fixed:
            mutable &= ~self.select(fixed)
        if not mutable.any():
            raise ValueError(f"Nothing left to design in {self.name}")
        return mutable

    def native_sequences(self) -> dict[str, str]:
        """Native one-letter sequence per protein chain, as the model sees it.

        Modified residues map to their standard parent; ``X`` if there is none.
        """
        from artios.common.residue_names import PROTEIN_COMMON_THREE_TO_ONE

        out: dict[str, list[str]] = {}
        for r in self.residues:
            if r is not None:
                out.setdefault(r.chain, []).append(
                    PROTEIN_COMMON_THREE_TO_ONE.get(r.canonical_res_name, "X")
                )
        return {c: "".join(s) for c, s in out.items()}


def load_target(
    path: str | Path,
    *,
    name: str | None = None,
    model: int = 1,
    chains: Iterable[str] | None = None,
    remove_crystallization_aids: bool = True,
    remove_glycans: bool = True,
) -> DesignTarget:
    """Load a PDB or mmCIF file (optionally gzipped) as a ``DesignTarget``.

    Uses the asymmetric unit (or the single model) as given. Only atoms present
    in the file are used: missing residues are left out, not modelled, so they
    are neither designed nor part of the neighbour graph.

    Args:
        path: ``.pdb``, ``.cif`` or ``.mmcif`` (``.gz`` allowed for mmCIF).
        name: Target name used in outputs. Default: file stem.
        model: 1-based model index for multi-model files.
        chains: Keep only these author chain IDs. Ligands and ions carry the
            author chain they were deposited with and are kept or dropped with
            it.
        remove_crystallization_aids: Drop common additives (SO4, GOL, PEG,
            ...), artifact ligands and unbound metal ions, as in training.
        remove_glycans: Drop glycan residues, as in training.
    """
    path = Path(path)
    name = name or path.name.split(".")[0]
    suffixes = [s.lower() for s in path.suffixes]
    if ".pdb" in suffixes or ".ent" in suffixes:
        with tempfile.TemporaryDirectory() as tmp:
            cif_path = Path(tmp) / f"{name}.cif"
            _pdb_to_cif_file(path, model=model).write(cif_path)
            reader = MmcifReader(cif_path)
            atom_array = reader.atom_array()
    else:
        reader = MmcifReader(path)
        atom_array = reader.atom_array(model=model)

    if chains is not None:
        chains = list(chains)
        missing = set(chains) - set(atom_array.auth_asym_id)
        if missing:
            raise ValueError(f"Chains {sorted(missing)} not found in {path}")
        atom_array = atom_array[np.isin(atom_array.auth_asym_id, chains)]
    entity_poly_type = reader.header.entity_poly_type
    atom_array = filter_module.clean_structure(
        atom_array,
        entity_poly_type=entity_poly_type,
        remove_unknown_residues=False,
        remove_distant_ca_chains=False,
        remove_crystallization_aids=remove_crystallization_aids,
        remove_artifact_ligands=remove_crystallization_aids,
        remove_unbound_metal_ions=remove_crystallization_aids,
    )
    if atom_array is None or len(atom_array) == 0:
        raise ValueError(f"No atoms left in {path} after cleaning")
    atom_array = residue_repair.fix_arginine(atom_array)
    atom_array = residue_repair.mse_to_met(atom_array)
    if remove_glycans:
        atom_array = atom_array[~np.isin(atom_array.res_name, list(ccd_sets.GLYCANS))]
    atom_array.set_annotation("is_resolved", np.ones(len(atom_array), dtype=bool))
    atom_array = struct_annotation.populate_ccd_reference(atom_array)

    structure = AnnotatedStructure.annotate(
        atom_array,
        sequences=reader.sequences_for(atom_array),
        entity_poly_type=entity_poly_type,
        include_ref_conformer=True,
    )
    n_atom = len(structure.atoms)
    zeros = np.zeros(n_atom, dtype=int)
    structure = structure.with_motif(
        MotifAnnotation(is_motif=zeros, is_motif_related=zeros)
    ).without_non_motif_sidechains()
    if not (structure.type_masks.is_protein > 0).any():
        raise ValueError(f"{path} contains no protein chain")

    return DesignTarget(name=name, structure=structure, residues=_residues(structure))


_PROTEIN_BACKBONE = frozenset({"N", "CA", "C"})
_NUCLEIC_BACKBONE = frozenset({"P", "O5'", "C5'"})
_PEPTIDE_BOND_MAX = 2.0  # Å, C(i)-N(i+1)


def _pdb_to_cif_file(path: Path, model: int = 1) -> pdbx.CIFFile:
    """Convert a PDB file to an mmCIF that ``MmcifReader`` parses faithfully.

    Author chain IDs, residue numbers and insertion codes are kept as the
    ``auth_*`` fields. Each HETATM ligand residue gets its own label chain, as
    in deposited mmCIFs; HETATM residues peptide-bonded into a chain (MSE,
    SEP, ...) stay in it. ``label_seq_id`` follows the author numbering, so
    missing residues remain gaps. Chains with identical residue sequences
    share an entity, so homo-oligomer copies are recognised as such.
    """
    atoms = pdb.PDBFile.read(str(path)).get_structure(
        model=model, altloc="first", extra_fields=["b_factor", "occupancy", "charge"]
    )
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    n_res = len(starts) - 1
    res_of_atom = np.repeat(np.arange(n_res), np.diff(starts))
    names = [set(atoms.atom_name[a:b]) for a, b in zip(starts[:-1], starts[1:])]
    first = starts[:-1]

    def atom_coord(r: int, name: str):
        hit = np.flatnonzero(atoms.atom_name[starts[r] : starts[r + 1]] == name)
        return atoms.coord[starts[r] + hit[0]] if len(hit) else None

    def peptide_bonded(r: int, s: int) -> bool:
        c, n = atom_coord(r, "C"), atom_coord(s, "N")
        return (
            c is not None
            and n is not None
            and np.linalg.norm(c - n) < _PEPTIDE_BOND_MAX
        )

    has_backbone = [
        bool(_PROTEIN_BACKBONE <= n or _NUCLEIC_BACKBONE <= n) for n in names
    ]
    polymer = [not atoms.hetero[f] for f in first]
    for r in range(n_res):  # modified residues bonded into a chain
        if atoms.hetero[first[r]] and has_backbone[r]:
            same_chain = lambda s: atoms.chain_id[first[s]] == atoms.chain_id[first[r]]  # noqa: E731
            polymer[r] = (r > 0 and same_chain(r - 1) and peptide_bonded(r - 1, r)) or (
                r + 1 < n_res and same_chain(r + 1) and peptide_bonded(r, r + 1)
            )

    label_asym = np.empty(n_res, dtype=object)
    label_seq = np.full(n_res, ".", dtype=object)
    asym_count = 0
    poly_seqs: dict[str, list[int]] = {}
    for chain in dict.fromkeys(atoms.chain_id[first]):
        in_chain = [r for r in range(n_res) if atoms.chain_id[first[r]] == chain]
        poly = [r for r in in_chain if polymer[r]]
        if poly:
            asym_count += 1
            asym = int_to_letters(asym_count)
            seq_id, prev = 0, None
            for r in poly:
                num = int(atoms.res_id[first[r]])
                step = 1 if prev is None or num <= prev else num - prev
                seq_id, prev = seq_id + step, num
                label_asym[r], label_seq[r] = asym, str(seq_id)
            poly_seqs[asym] = poly
        for r in in_chain:
            if not polymer[r]:
                asym_count += 1
                label_asym[r] = int_to_letters(asym_count)

    entity_of_asym: dict[str, str] = {}
    entity_poly_type: dict[str, str] = {}
    entity_by_key: dict[tuple, str] = {}
    for asym in dict.fromkeys(label_asym):
        res = [r for r in range(n_res) if label_asym[r] == asym]
        if asym in poly_seqs:
            key = ("poly", *(str(atoms.res_name[first[r]]) for r in res))
        else:
            key = ("ligand", str(atoms.res_name[first[res[0]]]))
        if key not in entity_by_key:
            entity_by_key[key] = str(len(entity_by_key) + 1)
            if key[0] == "poly":
                protein = sum(_PROTEIN_BACKBONE <= names[r] for r in res) * 2 >= len(
                    res
                )
                entity_poly_type[entity_by_key[key]] = (
                    "polypeptide(L)" if protein else _nucleic_type(key[1:])
                )
        entity_of_asym[asym] = entity_by_key[key]

    atoms.set_annotation(
        "label_entity_id",
        np.array([entity_of_asym[label_asym[r]] for r in res_of_atom]),
    )
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atoms, data_block=path.stem)
    block = cif.block
    atom_site = block["atom_site"]
    atom_site["label_asym_id"] = pdbx.CIFColumn(label_asym[res_of_atom].astype(str))
    atom_site["label_seq_id"] = pdbx.CIFColumn(label_seq[res_of_atom].astype(str))
    block["entity_poly"] = pdbx.CIFCategory(
        {
            "entity_id": list(entity_poly_type),
            "type": list(entity_poly_type.values()),
        }
    )
    return cif


def _nucleic_type(res_names: tuple[str, ...]) -> str:
    from artios.common.constants import DNA_STD_RESIDUES

    dna = sum(n in DNA_STD_RESIDUES for n in res_names) * 2 >= len(res_names)
    return "polydeoxyribonucleotide" if dna else "polyribonucleotide"


def _residues(structure: AnnotatedStructure) -> tuple[Residue | None, ...]:
    atoms = structure.atoms
    centre = structure.centre_atom_indices
    is_protein = structure.type_masks.chain_mol_type[centre] == "protein"
    ins = atoms.ins_code if "ins_code" in atoms.get_annotation_categories() else None
    canonical = structure.derived.cano_seq_resname
    out: list[Residue | None] = []
    for i, prot in zip(centre, is_protein):
        if not prot:
            out.append(None)
            continue
        out.append(
            Residue(
                chain=str(atoms.auth_asym_id[i]),
                number=int(atoms.auth_seq_id[i]),
                ins_code=str(ins[i]).strip().upper() if ins is not None else "",
                res_name=str(atoms.res_name[i]),
                canonical_res_name=str(canonical[i]),
            )
        )
    return tuple(out)


def batch_contexts(contexts: Sequence[FullContext]) -> FullContext:
    """Pad and stack contexts with the same settings as the inference dataloader."""
    padding = data_configs["padding"]
    collate = CollateContexts(
        token_bin_ends=list(padding["token_bin_ends"].value),
        atom_bin_ends=list(padding["atom_bin_ends"].value),
        crop_size=384,
        motif_training=False,
        max_motif_residues=10,
    )
    return collate(list(contexts))
