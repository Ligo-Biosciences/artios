"""

Simple modulate function for absorbing AdaLN into the DiT block.

Do we combine diffusion transformer and transfomer?

Deepmind's pattern is one custom class and one custom forward for every specific function. There are no if else conditional flows. This is the pattern
we follow.

Cross attention and self attention are separated.

"""

from torch import Tensor, nn

from protenix.model.modules.components import Attention, Transition
from protenix.model.modules.primitives import LayerNorm


class DiTBlock(nn.Module):
    """Diffusion Transformer Block."""

    def __init__(self, c_token: int, num_heads: int):
        assert c_token % num_heads == 0, "Input dim should be divisible by num heads."
        self.attn_layernorm = LayerNorm(c_token, elementwise_affine=False)
        # Attention
        self.attention = Attention(
            q_dim=c_token,
            kv_dim=c_token,
            head_dim=c_token // num_heads,
            num_heads=num_heads,
            gating=True,
            final_linear_init="default",
        )
        self.mlp_layernorm = LayerNorm(c_token, elementwise_affine=False)
        # Transition
        self.transition = Transition(c_in=c_token, n=2, output_linear_init="default")

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
                [..., num_heads, N, N]
            pair_mask (Tensor): Pairwise attention mask.
                [..., N, N]
        """
        # Adaln + Attention
        attn_input = scale_attn * self.attn_layernorm(a) + shift_attn
        attn_output = self.attention(
            q_x=attn_input, kv_x=attn_input, pair_bias=pair_bias, pair_mask=pair_mask
        )
        a = a + (gate_attn * attn_output)

        # Adaln + MLP
        mlp_input = scale_mlp * self.mlp_layernorm(a) + shift_mlp
        mlp_output = self.transition(mlp_input)
        a = a + (gate_mlp * mlp_output)
        return a


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
