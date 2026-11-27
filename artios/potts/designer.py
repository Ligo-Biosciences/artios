"""Design and score sequences with ArtiosPotts.

    from artios.potts import PottsDesigner, load_target, parse_selection

    designer = PottsDesigner()
    target = load_target("complex.pdb")
    designs = designer.design(target, design_chains=["B"], num_sequences=8, seed=0)
    scores = designer.score(target, [d.sequences for d in designs])

The Potts fields ``h`` and couplings ``J`` depend only on the backbone, so each
call computes them once per target and samples every sequence from them.
"""

from __future__ import annotations

import dataclasses
import logging
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from artios.assets import POTTS_CHECKPOINT, download
from artios.common.constants import STD_RESIDUES_AND_GAP
from artios.common.residue_names import (
    PROTEIN_COMMON_ONE_TO_THREE,
    PROTEIN_COMMON_THREE_TO_ONE,
)
from artios.config import parse_configs
from artios.configs.configs_base import configs as configs_base
from artios.configs.configs_data import data_configs
from artios.configs.configs_inference import inference_configs
from artios.data.features import FullContext
from artios.model.artios_potts import ArtiosPotts, PottsParams
from artios.model.modules.potts import (
    compute_potts_energy,
    compute_potts_single_logits,
)
from artios.potts.structure import DesignTarget, ResidueSelector, batch_contexts

logger = logging.getLogger(__name__)

_IDX_TO_ONE = {
    idx: PROTEIN_COMMON_THREE_TO_ONE.get(name, "X")
    for name, idx in STD_RESIDUES_AND_GAP.items()
}
_STANDARD_AA = torch.tensor(
    sorted(STD_RESIDUES_AND_GAP[t] for t in PROTEIN_COMMON_ONE_TO_THREE.values())
)
_ONE_TO_IDX = {
    **{o: STD_RESIDUES_AND_GAP[t] for o, t in PROTEIN_COMMON_ONE_TO_THREE.items()},
    "X": STD_RESIDUES_AND_GAP["UNK"],
}


@dataclass(frozen=True)
class Design:
    """One sampled sequence for a target."""

    target: str
    seed: int
    index: int
    sequences: dict[str, str]
    """Full sequence per protein chain, designed and fixed positions alike."""
    designed: dict[str, str]
    """Per chain, ``"1"`` where the position was designed and ``"0"`` where fixed."""
    energy: float
    """Potts energy summed over designed positions (lower is better)."""
    residue_energy: tuple[float, ...]
    """Per designed position, in chain order."""
    native_identity: float | None
    """Fraction of designed positions matching the input structure's residues
    (``None`` for ArtiosBackbone outputs, which have no native sequence)."""


@dataclass(frozen=True)
class Score:
    """Model score of a given sequence on a target."""

    target: str
    energy: float
    """Potts energy summed over all protein positions."""
    pseudo_log_likelihood: float
    """Mean log p(s_i | rest) over the 20 standard amino acids, at positions
    holding one (``X`` positions are skipped)."""
    residue_energy: tuple[float, ...]
    residue_log_likelihood: tuple[float, ...]
    """Per protein position holding a standard amino acid."""


class PottsDesigner:
    """ArtiosPotts loaded once for repeated design and scoring.

    Args:
        checkpoint: Path to a checkpoint. Default: the released weights.
        device: Torch device. Default: CUDA if available.
        sampling: Overrides for ``artios_mpnn.potts.sampling`` config keys,
            e.g. ``{"temperature": 0.2, "exclude_cys": False}``.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | torch.device | None = None,
        sampling: Mapping[str, object] | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "mps":
            # Large targets abort the process in a Metal gather kernel.
            raise ValueError("ArtiosPotts does not run on Apple MPS; use the CPU")
        _check_sampling_keys(sampling or {})
        overrides = " ".join(
            f"--artios_mpnn.potts.sampling.{k} {shlex.quote(str(v))}"
            for k, v in (sampling or {}).items()
        )
        configs = {**configs_base, **{"data": data_configs}, **inference_configs}
        self.configs = parse_configs(
            configs=configs,
            arg_str=f"--model_type potts {overrides}",
            fill_required_with_null=True,
        )
        self.model = ArtiosPotts(self.configs).to(self.device)
        path = Path(checkpoint) if checkpoint else download(POTTS_CHECKPOINT)
        state = torch.load(path, self.device, weights_only=True)["model"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=self.configs.load_strict)
        self.model.eval()

    @torch.no_grad()
    def potts_params(
        self, target: DesignTarget, seed: int = 0
    ) -> tuple[PottsParams, FullContext]:
        """Ensemble-averaged h/J of ``target`` (no masks or biases applied)."""
        context = batch_contexts([target.build_context()]).to(self.device)
        generator = torch.Generator(self.device).manual_seed(seed)
        return self.model.compute_potts_params(context, generator), context

    @torch.no_grad()
    def design(
        self,
        target: DesignTarget,
        *,
        design_chains: Sequence[str] | None = None,
        fixed: Sequence[ResidueSelector] = (),
        redesign: Sequence[ResidueSelector] | None = None,
        num_sequences: int = 1,
        batch_size: int = 8,
        seed: int = 0,
    ) -> list[Design]:
        """Sample ``num_sequences`` sequences for ``target``.

        Results are reproducible for a given ``seed`` and ``batch_size``.
        """
        unbatched = target.build_context(
            design_chains=design_chains, fixed=fixed, redesign=redesign
        )
        mutable = torch.from_numpy(
            target.mutable_mask(
                design_chains=design_chains, fixed=fixed, redesign=redesign
            )
        )
        generator = torch.Generator(self.device).manual_seed(seed)
        params = self.model.compute_potts_params(
            batch_contexts([unbatched]).to(self.device), generator
        )
        designs: list[Design] = []
        for start in range(0, num_sequences, batch_size):
            n = min(batch_size, num_sequences - start)
            context = batch_contexts([unbatched] * n).to(self.device)
            output = self.model.sample_from_params(
                context, _repeat(params, n), generator
            )
            for b in range(n):
                designs.append(
                    self._decode(target, output, b, mutable, seed, start + b)
                )
        return designs

    @torch.no_grad()
    def score(
        self,
        target: DesignTarget,
        sequences: Sequence[Mapping[str, str] | str],
        seed: int = 0,
        batch_size: int = 64,
    ) -> list[Score]:
        """Score sequences on ``target``'s backbone.

        Each sequence is a ``{chain: sequence}`` mapping, or a single string
        for single-chain targets. Chains left out keep their native sequence.
        h and J are computed once; sequences are scored ``batch_size`` at a time.
        """
        params, context = self.potts_params(target, seed=seed)
        # Non-protein tokens (ligands, ions) keep their own token identity.
        native = context.token.restype[0].argmax(-1).cpu()
        encoded = [self._encode(target, s, native) for s in sequences]
        scores: list[Score] = []
        for start in range(0, len(encoded), batch_size):
            seqs = torch.stack(encoded[start : start + batch_size]).to(self.device)
            scores += self._score_batch(target, params, seqs)
        return scores

    def _score_batch(
        self, target: DesignTarget, params: PottsParams, seqs: torch.Tensor
    ) -> list[Score]:
        n_token = len(target.residues)
        is_protein = torch.from_numpy(target.is_protein)
        n = len(seqs)
        h, J = params.h.expand(n, -1, -1), params.J.expand(n, -1, -1, -1, -1)
        edge_index = params.edge_index.expand(n, -1, -1)
        edge_mask = params.edge_mask.expand(n, -1, -1)
        energy = compute_potts_energy(seqs, h, J, edge_index, edge_mask)[:, :n_token]
        logits = compute_potts_single_logits(seqs, h, J, edge_index, edge_mask)
        aa = _STANDARD_AA.to(self.device)
        log_p = torch.log_softmax(logits[..., aa], dim=-1)
        pos = (seqs[..., None] == aa).float().argmax(-1)
        ll = torch.gather(log_p, -1, pos[..., None]).squeeze(-1)[:, :n_token]
        standard = torch.isin(seqs[:, :n_token], aa).cpu()
        scores = []
        for b in range(n):
            e = energy[b][is_protein].cpu()
            l = ll[b].cpu()[is_protein & standard[b]]  # noqa: E741
            scores.append(
                Score(
                    target=target.name,
                    energy=float(e.sum()),
                    pseudo_log_likelihood=float(l.mean()),
                    residue_energy=tuple(e.tolist()),
                    residue_log_likelihood=tuple(l.tolist()),
                )
            )
        return scores

    @staticmethod
    def _encode(
        target: DesignTarget,
        sequence: Mapping[str, str] | str,
        token_restype: torch.Tensor,
    ) -> torch.Tensor:
        chains = target.protein_chains
        if isinstance(sequence, str):
            if len(chains) != 1:
                raise ValueError(
                    f"{target.name} has chains {chains}; pass a {{chain: sequence}} mapping"
                )
            sequence = {chains[0]: sequence}
        native = target.native_sequences()
        unknown = set(sequence) - set(chains)
        if unknown:
            raise ValueError(
                f"Chains {sorted(unknown)} are not protein chains of {target.name}; "
                f"protein chains are {chains}"
            )
        sequence = {**native, **sequence}
        for c in chains:
            if len(sequence[c]) != len(native[c]):
                raise ValueError(
                    f"Chain {c} has {len(native[c])} residues, sequence has {len(sequence[c])}"
                )
        cursor = dict.fromkeys(chains, 0)
        idx = token_restype.clone()
        for i, r in enumerate(target.residues):
            if r is None:
                continue
            letter = sequence[r.chain][cursor[r.chain]].upper()
            cursor[r.chain] += 1
            if letter not in _ONE_TO_IDX:
                raise ValueError(f"Unsupported residue {letter!r} at {r.label}")
            idx[i] = _ONE_TO_IDX[letter]
        return idx

    @staticmethod
    def _decode(target, output, b, mutable, seed, index) -> Design:
        n_token = len(target.residues)
        seq_idx = output.sampled_sequence[b, :n_token].argmax(-1).cpu()
        energy = output.energy[b, :n_token].cpu()
        sequences: dict[str, list[str]] = {}
        designed: dict[str, list[str]] = {}
        for i, r in enumerate(target.residues):
            if r is None:
                continue
            sequences.setdefault(r.chain, []).append(
                _IDX_TO_ONE.get(int(seq_idx[i]), "X")
            )
            designed.setdefault(r.chain, []).append("1" if mutable[i] else "0")
        native = target.native_sequences()
        flat_design = "".join("".join(sequences[c]) for c in sequences)
        flat_native = "".join(native[c] for c in sequences)
        flat_mask = np.array([m == "1" for c in sequences for m in designed[c]])
        same = np.array([a == b for a, b in zip(flat_design, flat_native)])
        return Design(
            target=target.name,
            seed=seed,
            index=index,
            sequences={c: "".join(s) for c, s in sequences.items()},
            designed={c: "".join(m) for c, m in designed.items()},
            energy=float(energy[mutable].sum()),
            residue_energy=tuple(energy[mutable].tolist()),
            native_identity=float(same[flat_mask].mean())
            if target.has_native_sequence
            else None,
        )


def _check_sampling_keys(sampling: Mapping[str, object]) -> None:
    """Reject keys that are not ``artios_mpnn.potts.sampling`` options."""
    defaults = configs_base["artios_mpnn"]["potts"]["sampling"]
    valid, shown = set(), []
    for key, value in defaults.items():
        if isinstance(value, dict):
            valid |= {f"{key}.{sub}" for sub in value}
            shown.append(f"{key}.<{'|'.join(value)}>")
        else:
            valid.add(key)
            shown.append(key)
    unknown = sorted(set(sampling) - valid)
    if unknown:
        raise ValueError(
            f"Unknown sampling option {', '.join(unknown)}; options are "
            + ", ".join(shown)
        )


def _repeat(params: PottsParams, n: int) -> PottsParams:
    """Repeat batch-1 params ``n`` times along the batch dimension."""
    return dataclasses.replace(
        params,
        **{
            f.name: getattr(params, f.name).repeat(
                n, *([1] * (getattr(params, f.name).dim() - 1))
            )
            for f in dataclasses.fields(params)
            if getattr(params, f.name) is not None
        },
    )
