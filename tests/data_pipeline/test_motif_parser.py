"""Tests for ``artios.data.parsing.motif``.

TODO: motif parsing is currently untested. The old coverage (a single
``FileNotFoundError`` smoke + unused fixtures) was removed during the
trivial-test pass. Fill in real tests before the next refactor of the motif
pipeline — candidates worth pinning:

- parsing a CCD-based motif spec (``tests/fixtures/motifs/motif_input.json``)
  produces a ``Motif`` with the expected residues / ligands.
- parsing a SMILES-based motif spec
  (``tests/fixtures/motifs/motif_input_smiles.json``) handles substructure
  matching correctly and populates atom maps.
- round-trip: ``Motif`` → spec dict → ``Motif`` (if that path exists).
"""
