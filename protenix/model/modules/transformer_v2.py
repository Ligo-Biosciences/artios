"""

Simple modulate function for absorbing AdaLN into the DiT block.

Do we combine diffusion transformer and transfomer?

Deepmind's pattern is one custom class and one custom forward for every specific function. There are no if else conditional flows. This is the pattern
we follow.

Cross attention and self attention are separated.

"""

from torch import Tensor, nn

from protenix.model.modules.components import Transition


class DiTBlock(nn.Module):
    """Diffusion Transformer Block."""

    def __init__(self, c_token: int, n_heads: int):
        # Attention
        # Transition
        self.transition = Transition(c_in=c_token, n=2, output_linear_init="zeros")
        pass

    def forward(
        self,
        *,
        a: Tensor,
        scale_attn: Tensor,
        shift_attn: Tensor,
        gate_attn: Tensor,
        scale_mlp: Tensor,
        shift_mlp: Tensor,
        gate_mlp: Tensor,
        pair_bias: Tensor,
        pair_mask: Tensor,
    ):
        """
        Forward pass of the transformer block.

        Args:
            a (Tensor): Input token representations.
                [..., c_token]
            scale_attn (Tensor): attention scaling.
                [..., c_token]
            shift_attn (Tensor): attention shift.
                [..., c_token]
            gate_attn (Tensor): attention gating.
                [..., c_token]
            scale_mlp (Tensor): MLP scaling.
                [..., c_token]
            shift_mlp (Tensor): MLP shift.
                [..., c_token]
            gate_mlp (Tensor): MLP gating.
                [..., c_token]
            pair_bias (Tensor): Pairwise attention bias.
                [..., N, N, no_heads]
            pair_mask (Tensor): Pairwise attention mask.
                [..., N, N]
        """
        # Attention + adaln
        # MLP + adaln
        # scaling and shift for both
        pass


class DiffusionTransformer(nn.Module):
    """"""

    def __init__(self):
        pass

    def forward(self):
        pass


class LocalAttnTransformerBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        # Attention
        # MLP
        pass


class LocalAttnTransformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        # Put the rearranges in here within the for loop.
        pass
