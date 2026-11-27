# artios.potts

This package holds the sequence-design API behind `artios design`, `artios score` and `artios params`. `structure.py` turns PDB, mmCIF and NPZ files into `DesignTarget`s, and `designer.py` holds `PottsDesigner`.

Keep these invariants when changing this package or the Potts model:

- **Outputs match the runner.** `PottsDesigner.design(num_sequences=1)` must give exactly the output of `ArtiosPotts.main_inference_loop`. NPZ targets must match the older runner's `MPNNDataset`. `tests/potts/test_designer.py` checks both.
- **h and J come from the backbone alone.** The fields `h` and couplings `J` depend only on the structure; the model sees no sequence. Compute them once per target (`compute_potts_params`) and sample every sequence from them (`sample_from_params`).
- **RNG order is part of reproducibility.** Changes to `artios/model/artios_potts.py` or `generator.py` must keep it, or say why not. Run `scripts/benchmark/potts_regression.py capture` before the change and `compare` after, on the same GPU.
- **Residues use author numbering.** `DesignTarget.residues` has one entry per model token and is `None` for anything not designable (ligands, ions, motif residues of scaffold outputs). Every mask in this package is indexed by token.
- **Errors name the fix.** User errors (bad selections, sequences or options) raise `ValueError` with a message that says what to change; the CLI turns them into exit code 2.

Run `uv run pytest tests/potts tests/test_potts.py` (about 2 minutes on CPU) and `uvx ruff@0.6.3 format .`.
