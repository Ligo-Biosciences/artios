from typing import Optional

import torch.nn as nn
from typing_extensions import assert_never

from protenix.data.features import FullContext
from protenix.model.modules.mpnn import MPNN
from protenix.model.output import ModelOutput
from protenix.model.utils import Mode


class ArtiosMPNN(nn.Module):
    """All-atom message passing neural network for sequence design."""

    def __init__(self, configs):
        super(ArtiosMPNN, self).__init__()

        self.train_zero_shot = configs.artios_mpnn.train_zero_shot
        self.sample_zero_shot = configs.artios_mpnn.sampling.zero_shot
        self.temp = configs.artios_mpnn.sampling.temperature
        self.smoothing_factor = configs.artios_mpnn.sampling.smoothing_factor

        self.mpnn = MPNN(**configs.artios_mpnn.model)

    def main_inference_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> ModelOutput:
        restype, _ = self.mpnn.sample(
            context=context,
            temperature=self.temp,
            zero_shot=self.sample_zero_shot,
            smoothing_factor=self.smoothing_factor,
            seed=seed,
        )

        return ModelOutput(restype=restype)

    def main_train_loop(
        self,
        context: FullContext,
        seed: Optional[int] = None,
    ) -> ModelOutput:
        """
        Main training loop for the all-atom MPNN model.

        Args:
            context (FullContext): Full context containing all features and labels.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            ModelOutput: Model output containing the prediction.
        """

        logits, _, _ = self.mpnn(
            context=context,
            zero_shot=self.train_zero_shot,
            seed=seed,
        )

        return ModelOutput(seq_logits=logits)

    def forward(
        self,
        context: FullContext,
        mode: Mode,
        *,
        seed: Optional[int] = None,
    ) -> ModelOutput:
        """
        Forward pass for all-atom MPNN module.

        Args:
            context (FullContext): Full context containing all features and labels.
            mode (Mode): Mode of operation.
            seed (Optional[int]): Seed for random number generation. Defaults to None.

        Returns:
            ModelOutput: Model output containing the prediction.
        """
        match mode:
            case Mode.TRAIN | Mode.EVAL:
                output = self.main_train_loop(
                    context=context,
                    seed=seed,
                )
            case Mode.INFERENCE:
                output = self.main_inference_loop(
                    context=context,
                    seed=seed,
                )
            case _:
                assert_never(mode)

        return output
