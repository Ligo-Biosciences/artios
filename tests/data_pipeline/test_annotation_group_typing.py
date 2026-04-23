"""Runtime typechecking contracts for ``_AnnotationGroup`` subclasses.

These tests fail loudly if:

- A field's jaxtyping dtype drifts from what the pipeline actually writes.
- A field's shape diverges from the ``n_atom`` contract (e.g. cross-field
  length mismatch within a group).
- A subclass forgets to declare ``FIELD_NAME``, breaking
  ``AnnotatedStructure.from_atom_array``'s ``cls(**groups)`` unpacking.
- An optional group's ``maybe_from_atom_array`` stops guarding
  partial-presence on the AtomArray.

We rely on ``TYPECHECK=1`` being the default in ``artios.utils.typing`` —
if it's ever disabled suite-wide, the dtype/shape tests would silently pass
since ``@typecheck`` becomes a no-op. The contract tests still fire.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from biotite.structure import AtomArray

from artios.data.structure.structure import (
    AnnotatedStructure,
    EntityIds,
    MotifAnnotation,
    ReferenceConformer,
    _AnnotationGroup,
)
from artios.utils.typing import TypeCheckError, should_typecheck

needs_typecheck = pytest.mark.skipif(
    not should_typecheck,
    reason="TYPECHECK=0 disables the @typecheck decorator, "
    "so dtype/shape violations would not raise.",
)


def _toy_atom_array(n: int) -> AtomArray:
    """Minimal AtomArray — we only need get_annotation_categories + set_annotation."""
    return AtomArray(n)


class TestDtypeEnforcement:
    @needs_typecheck
    def test_entity_ids_rejects_float_dtype(self):
        with pytest.raises(TypeCheckError):
            EntityIds(
                asym_id_int=np.zeros(5, dtype=np.float64),
                entity_id_int=np.zeros(5, dtype=np.int64),
                sym_id_int=np.zeros(5, dtype=np.int64),
            )

    @needs_typecheck
    def test_reference_conformer_rejects_int_ref_pos(self):
        """ref_pos is Float[..., 'n_atom 3']; int64 must fail."""
        with pytest.raises(TypeCheckError):
            ReferenceConformer(
                ref_pos=np.zeros((5, 3), dtype=np.int64),
                ref_charge=np.zeros(5, dtype=np.int64),
                ref_mask=np.zeros(5, dtype=np.int64),
                ref_space_uid=np.zeros(5, dtype=np.int64),
            )


class TestShapeEnforcement:
    @needs_typecheck
    def test_cross_field_length_mismatch_rejected(self):
        """All n_atom-shaped fields must bind the same length within one group."""
        with pytest.raises(TypeCheckError):
            EntityIds(
                asym_id_int=np.zeros(5, dtype=np.int64),
                entity_id_int=np.zeros(6, dtype=np.int64),
                sym_id_int=np.zeros(5, dtype=np.int64),
            )

    @needs_typecheck
    def test_ref_pos_rejects_wrong_trailing_dim(self):
        with pytest.raises(TypeCheckError):
            ReferenceConformer(
                ref_pos=np.zeros((5, 4), dtype=np.float64),  # must be (n, 3)
                ref_charge=np.zeros(5, dtype=np.int64),
                ref_mask=np.zeros(5, dtype=np.int64),
                ref_space_uid=np.zeros(5, dtype=np.int64),
            )


class TestSelfRegistration:
    def test_every_subclass_declares_field_name(self):
        """Every _AnnotationGroup subclass must set FIELD_NAME to a str."""
        for cls in _AnnotationGroup.__subclasses__():
            assert isinstance(
                getattr(cls, "FIELD_NAME", None), str
            ), f"{cls.__name__} is missing or has non-str FIELD_NAME"

    def test_field_name_corresponds_to_annotated_structure_field(self):
        """Every subclass's FIELD_NAME must name an AnnotatedStructure field —
        otherwise ``cls(**groups)`` in from_atom_array raises TypeError."""
        struct_fields = {f.name for f in dataclasses.fields(AnnotatedStructure)}
        bad = [
            cls.__name__
            for cls in _AnnotationGroup.__subclasses__()
            if cls.FIELD_NAME not in struct_fields
        ]
        assert not bad, (
            f"subclasses with unbound FIELD_NAME: {bad}; "
            f"known AnnotatedStructure fields: {sorted(struct_fields)}"
        )


class TestOptionalGroupPresence:
    def test_maybe_from_atom_array_none_when_all_absent(self):
        aa = _toy_atom_array(3)
        assert MotifAnnotation.maybe_from_atom_array(aa) is None

    def test_maybe_from_atom_array_raises_on_partial_presence(self):
        """If one field is present but not the other, refuse ambiguously."""
        aa = _toy_atom_array(3)
        aa.set_annotation("is_motif", np.ones(3, dtype=np.int64))
        # is_motif_related intentionally missing
        with pytest.raises(ValueError, match="is_motif_related"):
            MotifAnnotation.maybe_from_atom_array(aa)
