# Artios

Artios is a family of generative models from [Ligo Biosciences](https://ligo.bio) for designing ligand-binding proteins. Give it a catalytic or binding motif (a few residues plus one or more ligands) and it designs a protein around them.

The design pipeline has two models:

| Model | Task | Input | Output |
|---|---|---|---|
| **ArtiosBackbone** | All-atom motif scaffolding (EDM diffusion) | Motif CIF + ligand definitions (JSON) | Backbone structures (NPZ / CIF) |
| **ArtiosPotts** | Sequence design (Potts model on an MPNN encoder) | Backbone NPZ from ArtiosBackbone | Designed sequences (FASTA) |

Artios is built on [Protenix](https://github.com/bytedance/Protenix), ByteDance's open reproduction of AlphaFold 3 (see [NOTICE](NOTICE)).

> **TODO:** model weights on Hugging Face, paper/preprint link, headline results.

## Installation

Artios requires Python 3.11 or 3.12 and is developed on Linux with CUDA GPUs. It uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/Ligo-Biosciences/artios.git
cd artios
uv sync                 # core dependencies (inference)
uv sync --all-extras    # everything, including training and data preparation
```

Or with pip:

```bash
pip install -e .                 # core
pip install -e ".[train,data]"   # optional extras
```

| Extra | Needed for |
|---|---|
| `train` | Training (`runner/train_*.py`): wandb, neptune-scale, pandera |
| `data` | Training-data preparation (`scripts/data/`) and pocket annotation |
| `metrics` | Structure-quality and electrostatics metrics (`artios/metrics`, `artios/electrostatics`) |

### Chemical Component Dictionary (CCD)

Artios needs the wwPDB Chemical Component Dictionary and a pre-computed RDKit cache. Place both files in `ccd_files/` at the repository root:

```
ccd_files/
├── components.cif
└── components.cif.rdkit_mol.pkl
```

`scripts/data/gen_ccd_cache.py` builds the RDKit cache from `components.cif`.

> **TODO:** public download location for the CCD files.

## Usage

> **TODO:** quickstart command and Hugging Face weight download.

### Input format

Backbone generation takes a JSON file describing the design length and the motif:

```json
{
  "name": "cutinase",
  "task": "backbone",
  "spec": {
    "design": [{"length": 200}],
    "motif": {
      "cif_file": "evaluation/inputs/motifs/cutinase/cutinase_motif.cif",
      "ligands": [
        {
          "cif_res_name": "UNK",
          "chain_id": "1",
          "full_ligand": {
            "type": "smiles",
            "smiles": "CCCCCCC[C:1](=[O:2])[O:3]C1=CC2=C([C:4]=C1)C(=[C:5]C(=O)O2)C",
            "atom_map": {"C1": 1, "O1": 2, "O": 3, "C2": 4, "C3": 5}
          }
        }
      ]
    }
  }
}
```

Ligands can be given as CCD codes (`{"type": "ccd", "codes": ["ZN"]}`) or as atom-mapped SMILES, as above. See:

- [Motif input format](docs/motif_input_format.md): motif CIF rules, CCD and SMILES ligands
- [Sequence design input format](docs/sequence_design_input_format.md): fixed residues and backbone index maps
- [Motif stitching](docs/motif_stitching.md): how motif residues flow from backbone generation to the final design

Example motifs and inputs live in `evaluation/inputs/motifs/` and `evaluation/inputs/artios/`.

## Training

Training code is included as a reference: `runner/train_backbone.py` and `runner/train_potts.py`, launched through `scripts/training/`. See [preparing training data](docs/prepare_training_data.md) and [data sampling weights](docs/data_sampling_weights.md).

> **TODO:** training datasets on Hugging Face.

## Repository layout

```
artios/      model, data pipeline, metrics and utilities
configs/     default configuration for training and inference
runner/      inference and training entry points
scripts/     data preparation and training launch scripts
docs/        input formats and method notes
tests/       test suite (run with `uv run pytest -m "not slow"`)
```

## License

Artios is released under the [Apache License 2.0](LICENSE). It is a derivative of Protenix and includes code adapted from other open-source projects; see [NOTICE](NOTICE) for attributions.

## Citation

> **TODO:** citation.
