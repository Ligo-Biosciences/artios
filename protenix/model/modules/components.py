import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from protenix.model.modules.primitives import (
    BiasInitLinear,
    LayerNorm,
    Linear,
    LinearNoBias,
    attention,
)
from protenix.model.utils import flatten_final_dims, reshape_at_dim


class SwiGLU(nn.Module):
    """SwiGLU layer that computes swish(Linear(a)) * Linear(b)"""

    def __init__(self, c_in: int, c_out: int) -> None:
        """
        Args:
            c_in (int): Input channels
            c_out (int): Output channels
        """
        super(SwiGLU, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = LinearNoBias(c_in, c_out * 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        SwiGLU layer.

        Args:
            x (torch.Tensor): Input tensor
                [..., c_in]

        Returns:
            torch.Tensor: Output tensor
                [..., c_out]
        """
        a, b = self.linear(x).chunk(2, dim=-1)
        output = F.silu(a) * b
        return output


class AdaptiveLayerNorm(nn.Module):
    """
    Implements the Adaptive Layer Normlization introduced in arXiv:1911.07013
    Layers are initialized such that the conditioning has no impact at init
    i.e. the output values are 0.73 * LN(a) at init.
    """

    def __init__(self, *, c_a: int, c_s: int) -> None:
        """
        Args:
            c_a (int): the embedding dim of a(single feature aggregated atom info).
            c_s (int):  hidden dim [for single embedding].
        """
        super(AdaptiveLayerNorm, self).__init__()
        self.layernorm_a = LayerNorm(c_a, elementwise_affine=False)
        self.linear_s = BiasInitLinear(
            in_features=c_s, out_features=c_a, biasinit=1.0
        )  # gate values 0.73 @ init
        self.linear_nobias_s = LinearNoBias(
            in_features=c_s, out_features=c_a, init="zeros"
        )

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., c_a]
            s (torch.Tensor): single embedding
                [..., c_s]

        Returns:
            torch.Tensor: the updated a from AdaLN
                [..., c_a]
        """
        a = self.layernorm_a(a)
        a = torch.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)
        return a


class Transition(nn.Module):
    """
    Transition layer that computes LN -> SwiGLU -> LinearNoBias
    By default, this assumes the Transition is part of a residual block.
    """

    def __init__(self, *, c_in: int, n: int, output_linear_init: str = "zeros") -> None:
        """
        Args:
            c_in (int): the input dimension.
            n (int): factor by which c_in is multiplied to obtain hidden dimension.
            output_linear_init (str): the initialization method for the output linear layer.
                Defaults to "zeros".
        """
        super(Transition, self).__init__()
        self.n = n
        self.c_in = c_in
        self.output_linear_init = output_linear_init
        self.layernorm = LayerNorm(c_in)
        self.swiglu = SwiGLU(c_in=c_in, c_out=n * c_in)
        self.linear_no_bias = LinearNoBias(
            in_features=n * c_in, out_features=c_in, init=self.output_linear_init
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor
                [..., c]

        Returns:
            torch.Tensor: the output tensor as the same shape of x
                [..., c]
        """
        x = self.layernorm(x)
        x = self.linear_no_bias(self.swiglu(x))
        return x


class Attention(nn.Module):
    """
    Attention module. Handles QKV projections, gating and readout.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
        final_linear_init: str = "zeros",
    ) -> None:
        """Initialize attention module.

        Args:
            q_dim (int): Query input dimension
            kv_dim (int): Key/value input dimension
            head_dim (int): Dimension of each attention head
            num_heads (int): Number of attention heads
            gating (bool, optional): Whether to use gating. Defaults to True.
            final_linear_init (str, optional): Init scheme of final linear. Defaults to zeros.
        """
        super().__init__()

        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.gating = gating
        self.hidden_dim = num_heads * head_dim
        self.final_linear_init = final_linear_init

        # QKV initializations
        self.linear_q = Linear(q_dim, self.hidden_dim, bias=True)
        self.linear_k = LinearNoBias(kv_dim, self.hidden_dim)
        self.linear_v = LinearNoBias(kv_dim, self.hidden_dim)

        # Gating
        if self.gating:
            self.gate_linear = nn.Sequential(
                Linear(q_dim, self.hidden_dim, init="gating"), nn.Sigmoid()
            )

        # Output linear
        self.linear_output = LinearNoBias(
            self.hidden_dim, q_dim, init=final_linear_init
        )

    def forward(
        self,
        q_x: Tensor,
        kv_x: Tensor,
        pair_bias: Tensor,
        pair_mask: Tensor,
    ):
        """
        Args:
            q_x: input from which the queries will be derived from
                [..., N_q, q_dim]
            kv_x: input from which the key and values will be derived from
                [..., N_kv, kv_dim]
            pair_bias: per-head pair biases
                [..., num_heads, N_q, N_kv]
            pair_mask: pairwise attention mask
                [..., N_q, N_kv]

        Returns:
            torch.Tensor: Output tensor
                [..., N_q, q_dim]
        """
        assert pair_bias.shape[-3] == self.num_heads
        assert pair_bias.shape[-2:] == (q_x.shape[-2], kv_x.shape[-2])
        assert pair_mask.shape[-2:] == (q_x.shape[-2], kv_x.shape[-2])

        # Project QKV
        q = reshape_at_dim(
            self.linear_q(q_x), dim=-1, target_shape=(self.num_heads, self.head_dim)
        )  # [..., n_q, n_heads, c]
        k = reshape_at_dim(
            self.linear_k(kv_x), dim=-1, target_shape=(self.num_heads, self.head_dim)
        )  # [..., n_kv, n_heads, c]
        v = reshape_at_dim(
            self.linear_v(kv_x), dim=-1, target_shape=(self.num_heads, self.head_dim)
        )  # [..., n_kv, n_heads, c]

        q = q.transpose(-2, -3)  # [..., n_heads, n_q, c]
        k = k.transpose(-2, -3)  # [..., n_heads, n_kv, c]
        v = v.transpose(-2, -3)  # [..., n_heads, n_kv, c]

        # Add head dim
        pair_mask = pair_mask.unsqueeze(dim=-3)  # [..., n_heads = 1, n_q, n_kv]

        attn_out = attention(
            q=q, k=k, v=v, pair_mask=pair_mask, attn_bias=pair_bias, inf=1e9
        )  # [..., n_heads, n_q, c]

        attn_out = attn_out.transpose(-2, -3)  # [..., n_q, n_heads, c]
        attn_out = flatten_final_dims(attn_out, num_dims=2)  # [..., n_q, n_heads * c]

        # Gating
        if self.gating:
            gate = self.gate_linear(q_x)  # [..., n_q, n_heads * c]
            attn_out = gate * attn_out  # [..., n_q, n_heads * c]

        # Final projection
        attn_out = self.linear_output(attn_out)  # [..., n_q, q_dim]
        return attn_out
