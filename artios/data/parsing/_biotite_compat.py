"""Thin adapter over biotite's private mmCIF API.

The mmCIF pipeline uses a handful of underscore-prefixed functions on
``biotite.structure.io.pdbx.convert``: ``_filter_model``,
``_fill_annotations``, ``_parse_inter_residue_bonds``, ``_get_box``,
``_get_transformations``, ``_parse_operation_expression``,
``_apply_transformations``. These are private and biotite can change
their signatures across releases — in fact already has
(``_filter_model(atom_site, model_starts, model)`` → ``(atom_site, model)``).

Corralling the private access into one module means a biotite upgrade
is a one-file patch instead of a scavenger hunt. Callers should import
the public re-exports here and never reach into ``pdbx.convert``
directly.

TODO: bonds-level private access (``BondList._bonds``,
``_remove_redundant_bonds``, ``_get_max_bonds_per_atom``) is still
scattered across ``filter.py`` / ``annotation_v2.py``. Folding it in
here is a follow-up.
"""

from __future__ import annotations

from biotite.structure.io.pdbx import convert as _pdbx_convert

# Match the legacy-file behaviour of ignoring metal-coordinate bonds in
# _struct_conn. Module-load-time side effect; safe because the table is
# shared state across the biotite module.
_pdbx_convert.PDBX_BOND_TYPE_ID_TO_TYPE.pop("metalc", None)


filter_model = _pdbx_convert._filter_model
"""Pick a single model's rows out of an ``atom_site`` category.

Thin re-export of ``pdbx.convert._filter_model``.
Signature: ``filter_model(atom_site, model) -> atom_site_slice``.
"""

fill_annotations = _pdbx_convert._fill_annotations
"""Populate an ``AtomArray``'s annotations from an ``atom_site`` category.

Thin re-export of ``pdbx.convert._fill_annotations``.
Signature: ``fill_annotations(atoms, atom_site, extra_fields, use_author_fields)``.
"""

parse_inter_residue_bonds = _pdbx_convert._parse_inter_residue_bonds
"""Extract ``struct_conn`` bonds from the CIF block.

Thin re-export of ``pdbx.convert._parse_inter_residue_bonds``.
Signature: ``parse_inter_residue_bonds(atom_site, struct_conn) -> BondList``.
"""

get_box = _pdbx_convert._get_box
"""Extract the crystallographic box from a CIF block.

Thin re-export of ``pdbx.convert._get_box``.
Signature: ``get_box(block) -> np.ndarray | None``.
"""

get_transformations = _pdbx_convert._get_transformations
"""Parse ``_pdbx_struct_oper_list`` into a dict of transform matrices.

Thin re-export of ``pdbx.convert._get_transformations``.
Signature: ``get_transformations(struct_oper_category) -> dict``.
"""

parse_operation_expression = _pdbx_convert._parse_operation_expression
"""Parse an ``oper_expression`` string (e.g. ``"1,2,3"`` or ``"(1-5)"``).

Thin re-export of ``pdbx.convert._parse_operation_expression``.
Signature: ``parse_operation_expression(expr) -> list[list[str]]``.
"""

apply_transformations = _pdbx_convert._apply_transformations
"""Apply an expression of transformations to a sub-structure.

Thin re-export of ``pdbx.convert._apply_transformations``.
Signature:
``apply_transformations(sub_structure, transformations, operations) -> AtomArray``.
"""
