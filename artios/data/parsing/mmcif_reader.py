"""Typed, read-only wrapper around a biotite ``pdbx.CIFFile``.

Step 3 of the cif.py refactor. ``MmcifReader`` exposes the header-level
metadata and the lightweight ``atom_array()`` parse that inference needs
via ``Motif.from_spec``. It does **not** expand bioassemblies, does not
run filters, and does not touch annotations — those belong to
``load_bioassembly`` in ``cif.py`` (added in a later step).

Design notes:

- The header is materialised eagerly into a frozen ``MmcifHeader``
  dataclass; reading costs are trivial next to bioassembly expansion, and
  eager materialisation means callers can drop the reader once they have
  what they need.
- ``atom_array`` mirrors the legacy ``MMCIFParser.get_structure``
  byte-for-byte. Refactoring the internals (notably, collapsing the
  biotite private-API calls behind an adapter) is step 6. For now we just
  lift the body so ``MMCIFParser`` can delegate instead of owning it.
- ``MMCIFParser`` still exists in ``cif.py`` as a shim that forwards
  ``get_structure`` to a shared module-level helper, so every legacy
  import keeps resolving — parity tests pin the behaviour.
"""

from __future__ import annotations

import functools
import gzip
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Union

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure import AtomArray

from artios.data.chem import ccd
from artios.data.parsing import _biotite_compat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MmcifHeader:
    """Header-level metadata extracted from an mmCIF file.

    - ``pdb_id``: lowercase PDB id from ``_entry.id``; empty string if
      the entry category is absent.
    - ``resolution``: first available value from
      ``refine.ls_d_res_high`` / ``em_3d_reconstruction.resolution`` /
      ``reflns.d_resolution_high``. ``-1.0`` when none are present
      (sentinel kept for back-compat with the legacy dict bag).
    - ``release_date``: ``yyyy-mm-dd`` date from
      ``pdbx_audit_revision_history``. Sentinel ``9999-12-31`` when
      absent (same rule the legacy code used — keeps sort order sane).
    - ``methods``: experimental methods from ``exptl.method``; empty
      tuple if absent.
    - ``entity_poly_type``: ``{label_entity_id: poly_type}`` from the
      ``entity_poly`` category. When that category is missing (e.g.
      PyMOL exports), the dict is populated via residue-name inference
      so downstream callers never have to special-case non-PDB CIFs.
    - ``sequences``: ``{label_entity_id: canonical_sequence}`` — derived
      from ``entity_poly_seq`` (with the ``MSE → MET`` substitution the
      rest of the pipeline assumes). Altloc filtering is skipped at this
      layer; pass an ``atom_array`` to ``MmcifReader.sequences_for`` to
      get the altloc-aware variant.
    """

    pdb_id: str
    resolution: float
    release_date: date
    methods: tuple[str, ...]
    entity_poly_type: dict[str, str]
    sequences: dict[str, str]


class MmcifReader:
    """Read-only typed access to an mmCIF file.

    Constructed from a path (gzipped or plain). All header fields are
    materialised eagerly on ``header``; ``atom_array()`` is the
    lightweight parse used by ``Motif.from_spec`` and the entry point
    ``load_bioassembly`` will call before annotation.
    """

    def __init__(self, path: Union[Path, str]) -> None:
        self._path = Path(path)
        self._cif = self._read(self._path)

    @staticmethod
    def _read(path: Path) -> pdbx.CIFFile:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                return pdbx.CIFFile.read(f)
        with open(path, "rt") as f:
            return pdbx.CIFFile.read(f)

    # -- raw access needed by callers still living on the old API --------

    @property
    def cif(self) -> pdbx.CIFFile:
        """The underlying biotite ``CIFFile``.

        Exposed so the legacy ``MMCIFParser`` (which still owns the
        heavy pipeline during the refactor) can reuse the same parsed
        object instead of re-reading the file.
        """
        return self._cif

    # -- header ----------------------------------------------------------

    @functools.cached_property
    def header(self) -> MmcifHeader:
        return MmcifHeader(
            pdb_id=self._pdb_id(),
            resolution=self._resolution(),
            release_date=self._release_date(),
            methods=self._methods(),
            entity_poly_type=self._entity_poly_type(),
            sequences=self._sequences(),
        )

    def _pdb_id(self) -> str:
        block = self._cif.block
        if "entry" not in block:
            return ""
        return block["entry"]["id"].as_item().lower()

    def _resolution(self) -> float:
        block = self._cif.block
        for category_item in (
            "refine.ls_d_res_high",
            "em_3d_reconstruction.resolution",
            "reflns.d_resolution_high",
        ):
            category, item = category_item.split(".")
            if category in block and item in block[category]:
                try:
                    value = block[category][item].as_array(float)[0]
                except ValueError:
                    continue
                if value == 0.0:
                    # biotite coerces "." → 0.0; treat as missing.
                    continue
                return float(value)
        return -1.0

    def _release_date(self) -> date:
        block = self._cif.block
        if "pdbx_audit_revision_history" in block:
            raw = block["pdbx_audit_revision_history"]["revision_date"].as_array()[0]
        else:
            raw = "9999-12-31"
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(
                f"Invalid release date {raw!r} in {self._path.name}: expected yyyy-mm-dd"
            ) from exc

    def _methods(self) -> tuple[str, ...]:
        block = self._cif.block
        if "exptl" not in block:
            return ()
        return tuple(block["exptl"]["method"].as_array())

    # -- polymer-entity metadata -----------------------------------------

    def _entity_poly_type(self) -> dict[str, str]:
        """``{entity_id: poly_type}`` with a fallback for non-PDB CIFs.

        When ``entity_poly`` is absent (e.g. PyMOL exports), the poly type
        is inferred by comparing ``atom_site`` residue names against the
        standard protein / DNA / RNA sets. This matches the legacy
        fallback behaviour so motif CIFs and hand-crafted inputs keep
        parsing.
        """
        block = self._cif.block
        if "entity_poly" in block:
            ep = block["entity_poly"]
            entity_ids = ep["entity_id"].as_array(str)
            types = ep["type"].as_array(str)
            return dict(zip(entity_ids, types))
        logger.warning(
            "%s has no 'entity_poly' category; inferring polymer types from residue names",
            self._path.name,
        )
        return self._infer_entity_poly_type_from_atom_site()

    def _infer_entity_poly_type_from_atom_site(self) -> dict[str, str]:
        # Lazy import avoids a cycle with artios.common.constants at import
        # time (constants.py depends on ccd.py which depends on biotite).
        from artios.common.constants import (
            DNA_STD_RESIDUES,
            PROT_STD_RESIDUES_ONE_TO_THREE,
            RNA_STD_RESIDUES,
        )

        atom_site = self._cif.block.get("atom_site")
        if atom_site is None:
            return {}
        entity_ids = atom_site["label_entity_id"].as_array(str)
        res_names = atom_site["label_comp_id"].as_array(str)
        out: dict[str, str] = {}
        for eid in np.unique(entity_ids):
            names = set(res_names[entity_ids == eid])
            if names <= set(PROT_STD_RESIDUES_ONE_TO_THREE.values()):
                out[eid] = "polypeptide(L)"
            elif names <= set(DNA_STD_RESIDUES.keys()):
                out[eid] = "polydeoxyribonucleotide"
            elif names <= set(RNA_STD_RESIDUES.keys()):
                out[eid] = "polyribonucleotide"
            elif names <= (set(DNA_STD_RESIDUES.keys()) | set(RNA_STD_RESIDUES.keys())):
                out[eid] = "polydeoxyribonucleotide/polyribonucleotide hybrid"
            # else: non-polymer entity (ligand) — intentionally absent.
        return out

    def _sequences(self, atom_array: AtomArray | None = None) -> dict[str, str]:
        """``{entity_id: canonical_sequence}``.

        Called twice during the full pipeline: once without an atom_array
        (cheap, reads entity_poly_seq straight through) and once with the
        parsed atom_array so altloc filtering picks the same residue
        variant that ended up in the structure. Mirrors the legacy
        ``get_sequences`` shape.
        """
        poly_res_names = self._poly_res_names(atom_array)
        return {
            eid: ccd.res_names_to_sequence(names)
            for eid, names in poly_res_names.items()
        }

    def sequences_for(self, atom_array: AtomArray) -> dict[str, str]:
        """Altloc-aware variant of ``header.sequences``.

        Use this once an ``atom_array`` is built so the canonical sequence
        reflects the actual residues the parser kept after altloc filter.
        """
        return self._sequences(atom_array)

    def _poly_res_names(self, atom_array: AtomArray | None) -> dict[str, list[str]]:
        entity_res_names: dict[str, dict[int, str]] = {}
        if atom_array is not None:
            res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=False)
            for s in res_starts:
                eid = atom_array.label_entity_id[s]
                entity_res_names.setdefault(eid, {})[int(atom_array.res_id[s])] = (
                    atom_array.res_name[s]
                )

        block = self._cif.block
        if "entity_poly_seq" not in block:
            # Fallback path for PyMOL exports.
            logger.warning(
                "%s has no 'entity_poly_seq' category; inferring sequence from atom_site",
                self._path.name,
            )
            return self._infer_poly_res_names_from_atom_site(entity_res_names)

        eps = block["entity_poly_seq"]
        all_entity_ids = eps["entity_id"].as_array(str)
        all_mon_ids = eps["mon_id"].as_array(str)
        all_mon_ids = np.where(all_mon_ids == "MSE", "MET", all_mon_ids)
        all_num = eps["num"].as_array(int)

        out: dict[str, list[str]] = {}
        for entity_id in self._entity_poly_type():
            mask = all_entity_ids == entity_id
            mon_ids = all_mon_ids[mask]
            nums = all_num[mask]
            if np.unique(nums).size == nums.size:
                out[entity_id] = mon_ids
                continue
            # Altloc residues — pick the variant that matches atom_array if
            # available, else keep the first.
            select = np.zeros(len(nums), dtype=bool)
            matching = nums[0]
            for i, rid in enumerate(nums):
                if rid != matching:
                    continue
                picked = entity_res_names.get(entity_id, {}).get(int(rid))
                if picked is None:
                    select[i] = True
                elif picked == mon_ids[i]:
                    select[i] = True
                if select[i]:
                    matching += 1
            mon_ids = mon_ids[select]
            nums = nums[select]
            assert len(nums) == max(nums)
            out[entity_id] = mon_ids
        return out

    def _infer_poly_res_names_from_atom_site(
        self, entity_res_names: dict[str, dict[int, str]]
    ) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for entity_id in self._entity_poly_type().keys():
            if entity_id not in entity_res_names:
                continue
            ordered = sorted(entity_res_names[entity_id].keys())
            names = [entity_res_names[entity_id][r] for r in ordered]
            names = ["MET" if n == "MSE" else n for n in names]
            out[entity_id] = np.array(names)
        return out

    # -- atom array (lightweight parse used by Motif.from_spec) ----------

    def atom_array(
        self,
        *,
        altloc: str = "first",
        model: int = 1,
        bond_length_threshold: float | None = 2.4,
    ) -> AtomArray:
        """Parse the asymmetric unit of the CIF into an ``AtomArray``.

        Mirrors the legacy ``MMCIFParser.get_structure`` body, including
        altloc filtering, ``auth_*`` fallbacks for CIFs exported by tools
        that skip those fields, and the two-pass inter-residue bond
        inference (once with auth-seq-id, once more after
        ``replace_auth_with_label`` rewrites ``res_id`` to ``label_seq_id``).

        Args:
            altloc: ``"first"`` / ``"all"`` / ``"global_largest"`` / an
                explicit altloc letter. See ``_filter_altloc``.
            model: 1-based model index.
            bond_length_threshold: Filter for ``struct_conn`` bonds whose
                inferred length exceeds the threshold (Å). ``None``
                disables the filter. Default ``2.4`` matches legacy.
        """
        atoms = _parse_asymmetric_unit(
            self._cif,
            model=model,
            bond_length_threshold=bond_length_threshold,
        )
        atoms = _filter_altloc(atoms, altloc=altloc)
        atoms = ccd.add_inter_residue_bonds(
            atoms,
            exclude_struct_conn_pairs=True,
            remove_far_inter_chain_pairs=True,
        )
        atoms = _replace_auth_with_label(atoms)
        atoms = ccd.add_inter_residue_bonds(atoms, exclude_struct_conn_pairs=True)
        return atoms


# ---------------------------------------------------------------------------
# Module-level helpers used by MmcifReader.atom_array (shared with the
# legacy MMCIFParser during migration — see cif.py).
# ---------------------------------------------------------------------------


def _parse_asymmetric_unit(
    cif: pdbx.CIFFile,
    *,
    model: int = 1,
    bond_length_threshold: float | None = 2.4,
) -> AtomArray:
    """Build an ``AtomArray`` for a single model from the ``atom_site``
    category. Includes intra-residue bonds and ``struct_conn`` links
    (filtered by the optional length threshold)."""
    block = cif.block
    atom_site = block["atom_site"]

    use_author_fields = True
    extra_fields = {"label_asym_id", "label_entity_id", "label_seq_id"}
    if "auth_asym_id" in atom_site:
        extra_fields.add("auth_asym_id")
    if "auth_seq_id" in atom_site:
        extra_fields.add("auth_seq_id")
    for atom_site_name, alt_name in (
        ("occupancy", "occupancy"),
        ("pdbx_formal_charge", "charge"),
        ("B_iso_or_equiv", "b_factor"),
        ("label_alt_id", "label_alt_id"),
    ):
        if atom_site_name in atom_site:
            extra_fields.add(alt_name)

    model_atom_site = _biotite_compat.filter_model(atom_site, model)
    model_length = model_atom_site.row_count
    atoms = AtomArray(model_length)
    atoms.coord[:, 0] = model_atom_site["Cartn_x"].as_array(np.float32)
    atoms.coord[:, 1] = model_atom_site["Cartn_y"].as_array(np.float32)
    atoms.coord[:, 2] = model_atom_site["Cartn_z"].as_array(np.float32)
    atoms.box = _biotite_compat.get_box(block)

    _biotite_compat.fill_annotations(
        atoms, model_atom_site, extra_fields, use_author_fields
    )

    # Fallback for tools (PyMOL etc.) that don't emit auth_* fields.
    if not hasattr(atoms, "auth_seq_id") or atoms.auth_seq_id is None:
        if hasattr(atoms, "label_seq_id"):
            logger.warning(
                "%s: missing 'auth_seq_id', copying from 'label_seq_id'", cif
            )
            atoms.set_annotation("auth_seq_id", atoms.label_seq_id.copy())
    if not hasattr(atoms, "auth_asym_id") or atoms.auth_asym_id is None:
        if hasattr(atoms, "label_asym_id"):
            logger.warning(
                "%s: missing 'auth_asym_id', copying from 'label_asym_id'", cif
            )
            atoms.set_annotation("auth_asym_id", atoms.label_asym_id.copy())

    bonds = struc.connect_via_residue_names(atoms, inter_residue=False)
    if "struct_conn" in block:
        conn_bonds = _biotite_compat.parse_inter_residue_bonds(
            model_atom_site, block["struct_conn"]
        )
        coord1 = atoms.coord[conn_bonds._bonds[:, 0]]
        coord2 = atoms.coord[conn_bonds._bonds[:, 1]]
        dist = np.linalg.norm(coord1 - coord2, axis=1)
        if bond_length_threshold is not None:
            conn_bonds._bonds = conn_bonds._bonds[dist < bond_length_threshold]
        bonds = bonds.merge(conn_bonds)
    atoms.bonds = bonds
    return atoms


def _filter_altloc(atom_array: AtomArray, altloc: str = "first") -> AtomArray:
    """Filter alternate-conformation atoms by strategy.

    Strategies:
    - ``"all"``: keep everything.
    - ``"first"``: pick the alphabetically-first altloc letter across the
      array, drop the rest. Atoms without an altloc ('.') always stay.
    - ``"global_largest"``: rank altloc letters by mean occupancy, then
      per (chain,res_id) keep the highest-ranked letter that appears at
      that residue.
    - explicit letter (``"A"``, ``"B"``, …): keep only that altloc plus
      '.' atoms.

    Ported from the legacy ``MMCIFParser.filter_altloc``.
    """
    from collections import defaultdict

    if altloc == "all":
        return atom_array

    if altloc == "first":
        letters = np.unique(atom_array.label_alt_id)
        if len(letters) == 1 and letters[0] == ".":
            return atom_array
        letters = letters[letters != "."]
        chosen = np.sort(letters)[0]
        return atom_array[np.isin(atom_array.label_alt_id, [chosen, "."])]

    if altloc == "global_largest":
        occ_dict: dict[str, list[float]] = defaultdict(list)
        res_altloc: dict[tuple[str, int], list[str]] = defaultdict(list)
        res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        for res_start, _res_end in zip(res_starts[:-1], res_starts[1:]):
            letter = atom_array.label_alt_id[res_start]
            if letter == ".":
                continue
            occ_dict[letter].append(atom_array.occupancy[res_start])
            key = (atom_array.chain_id[res_start], int(atom_array.res_id[res_start]))
            res_altloc[key].append(letter)
        ranked = [
            letter
            for letter, _ in sorted(
                ((letter, np.mean(occs)) for letter, occs in occ_dict.items()),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        mask = np.zeros(len(atom_array), dtype=bool)
        for res_start, res_end in zip(res_starts[:-1], res_starts[1:]):
            letter = atom_array.label_alt_id[res_start]
            if letter == ".":
                mask[res_start:res_end] = True
                continue
            key = (atom_array.chain_id[res_start], int(atom_array.res_id[res_start]))
            choices = [x for x in ranked if x in res_altloc[key]]
            if choices and choices[0] == letter:
                mask[res_start:res_end] = True
        return atom_array[mask]

    # Explicit altloc letter
    return atom_array[np.isin(atom_array.label_alt_id, [altloc, "."])]


def _replace_auth_with_label(atom_array: AtomArray) -> AtomArray:
    """Overwrite ``chain_id`` with ``label_asym_id`` and reset ligand
    ``res_id`` sequentially within each chain.

    Polymer chains get their ``label_seq_id`` copied into ``res_id`` as
    an integer; non-polymer chains (``label_seq_id == "."``) are
    renumbered 1..N within the chain so downstream code can index on a
    contiguous integer.

    Ported from the legacy ``MMCIFParser.replace_auth_with_label`` with
    one fix: ``res_id`` is built as an integer array from the start, so
    wide values (>9999) no longer silently truncate through a string
    dtype (bug-scan finding #2).
    """
    atom_array.chain_id = atom_array.label_asym_id.copy()
    n = len(atom_array)
    res_id = np.zeros(n, dtype=int)
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
    for chain_start, chain_stop in zip(chain_starts[:-1], chain_starts[1:]):
        if atom_array.label_seq_id[chain_start] != ".":
            # Polymer chain: label_seq_id is a stringified integer.
            seq_ids = atom_array.label_seq_id[chain_start:chain_stop]
            res_id[chain_start:chain_stop] = seq_ids.astype(int)
            continue
        # Non-polymer chain (ligand): reset per-residue res_id to 1..N.
        local = atom_array[chain_start:chain_stop]
        res_starts = struc.get_residue_starts(local, add_exclusive_stop=True)
        num = 1
        for r_start, r_stop in zip(res_starts[:-1], res_starts[1:]):
            res_id[chain_start + r_start : chain_start + r_stop] = num
            num += 1
    atom_array.res_id = res_id
    return atom_array
