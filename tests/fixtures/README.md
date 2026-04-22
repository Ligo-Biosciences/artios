# Test fixtures

Data consumed by the test suite. No Python code lives here except the
fixture regeneration script.

## Layout

```
fixtures/
├── structures/         # Golden AnnotatedStructure .npz (5 enzymes)
├── cif/                # Raw mmCIF inputs (biotite I/O tests)
├── parsing/            # 5s8i triple (.cif/.npz/.pkl.gz) — legacy parsing fixture
├── motifs/             # Motif input .json + .cif
└── regenerate.py       # Rebuild structures/*.npz from pdb-samples/enzyme/
```

## Golden structures

Five small enzyme structures pre-parsed into `AnnotatedStructure.npz`.
Named by their PDB ID so provenance is self-documenting.

| File | Size | Tokens | Atoms | Chains | Notes |
|---|---|---|---|---|---|
| `structures/2g1k.npz` | 39 KB | 188 | 1318 | 2 | Smallest — good for fast iteration |
| `structures/1m2g.npz` | 55 KB | 286 | 1978 | 3 | 3 chains with ligand + cofactor |
| `structures/3iwd.npz` | 53 KB | 310 | 2130 | 6 | Multiple cofactor chains |
| `structures/1dai.npz` | 71 KB | 480 | 3416 | 4 | Medium enzyme |
| `structures/4lbp.npz` | 69 KB | 440 | 3192 | 8 | Multi-chain: 8 chains |

All five are protein-dominated enzymes with some ligand atoms (substrate,
cofactor, or ion). None include DNA or RNA — add a nucleic-acid fixture
when that coverage is needed.

### Regenerating

Source CIFs are committed alongside the fixtures in `cif/` — the
regeneration script reads from there, so this directory is self-contained
and doesn't depend on any gitignored `pdb-samples/`. To rebuild after an
`AnnotatedStructure` change:

```sh
uv run python -m tests.fixtures.regenerate
```

The script parses each source CIF via `MMCIFParser`, calls
`AnnotatedStructure.from_atom_array(...)`, verifies the `.save/.load`
round-trip, and writes to `structures/<pdb_id>.npz`. Run it whenever the
serialization format or annotation pipeline changes — tests pin against
these files, so a silent contract break will surface as load failures
rather than ticking along on stale data.

## Other fixtures

| Path | Consumer | Purpose |
|---|---|---|
| `cif/1bn1.cif` | `tests/utils/test_file_io.py` | Raw CIF for biotite `AtomArray` / `AtomArrayStack` NPZ round-trip tests. |
| `parsing/5s8i.cif`, `.npz`, `.pkl.gz` | `tests/data_pipeline/test_structure.py`, `test_cropping.py` | Legacy multi-format fixture. Pre-dates the golden-sample pattern — keep until those tests migrate to `load_structure(...)`. |
| `motifs/*` | `tests/data_pipeline/test_motif_parser.py` | Motif parser inputs (`.json` specs + `.cif` template). |
