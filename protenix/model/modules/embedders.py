import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

# from protenix.model.modules.atom_attention import AtomAttentionEncoder
from protenix.model.modules.primitives import LinearNoBias
from protenix.utils.typing import Float, Int, typecheck


class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        """
        Args:
            r_max (int, optional): Relative position indices clip value. Defaults to 32.
            s_max (int, optional): Relative chain indices clip value. Defaults to 2.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        """
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.embed_relpos_features = LinearNoBias(
            in_features=(4 * self.r_max + 2 * self.s_max + 7),
            out_features=self.c_z,
            init="lecun",
        )
        self.input_feature = {
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
        }

    @typecheck
    def forward(
        self,
        *,
        asym_id: Int[Tensor, "*batch n_token"],
        residue_index: Int[Tensor, "*batch n_token"],
        entity_id: Int[Tensor, "*batch n_token"],
        sym_id: Int[Tensor, "*batch n_token"],
        token_index: Int[Tensor, "*batch n_token"],
    ) -> Float[Tensor, "*batch n_token n_token c_z"]:
        # Pairwise equality masks
        asym_i = rearrange(asym_id, "... n -> ... n 1")
        asym_j = rearrange(asym_id, "... n -> ... 1 n")
        same_chain = asym_i == asym_j

        residue_i = rearrange(residue_index, "... n -> ... n 1")
        residue_j = rearrange(residue_index, "... n -> ... 1 n")
        same_residue = residue_i == residue_j

        entity_i = rearrange(entity_id, "... n -> ... n 1")
        entity_j = rearrange(entity_id, "... n -> ... 1 n")
        same_entity = entity_i == entity_j

        # Relative residue position (clipped), with special index for different chains
        d_residue_clipped = (residue_i - residue_j + self.r_max).clip(0, 2 * self.r_max)
        different_chain_idx = 2 * self.r_max + 1
        d_residue = torch.where(same_chain, d_residue_clipped, different_chain_idx)
        rel_residue = F.one_hot(d_residue, num_classes=2 * self.r_max + 2)

        # Relative token position (clipped), with special index for different chain or residue
        token_i = rearrange(token_index, "... n -> ... n 1")
        token_j = rearrange(token_index, "... n -> ... 1 n")
        d_token_clipped = (token_i - token_j + self.r_max).clip(0, 2 * self.r_max)
        same_chain_and_residue = same_chain & same_residue
        d_token = torch.where(
            same_chain_and_residue, d_token_clipped, different_chain_idx
        )
        rel_token = F.one_hot(d_token, num_classes=2 * self.r_max + 2)

        # Relative chain position (sym_id difference), with special index for different entities
        sym_i = rearrange(sym_id, "... n -> ... n 1")
        sym_j = rearrange(sym_id, "... n -> ... 1 n")
        d_chain_clipped = (sym_i - sym_j + self.s_max).clip(0, 2 * self.s_max)
        different_entity_idx = 2 * self.s_max + 1
        d_chain = torch.where(same_entity, d_chain_clipped, different_entity_idx)
        rel_chain = F.one_hot(d_chain, num_classes=2 * self.s_max + 2)

        # Concatenate features and embed
        features = torch.cat(
            [rel_residue, rel_token, same_entity[..., None], rel_chain],
            dim=-1,
        ).float()
        return self.embed_relpos_features(features)


class FourierEmbedding(nn.Module):
    """
    Implements Algorithm 22 in AF3
    """

    def __init__(self, c: int, seed: int = 42) -> None:
        """
        Args:
            c (int): embedding dim.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        super(FourierEmbedding, self).__init__()
        self.c = c

        # Register buffers instead of parameters since they don't need gradients
        # This is more memory efficient and semantically clearer
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("w", torch.randn(c, generator=generator))
        self.register_buffer("b", torch.randn(c, generator=generator))

    def forward(self, t_hat_timestep: Tensor) -> Tensor:
        """
        Args:
            t_hat_timestep (Tensor): the noise level
                [..., N_sample]

        Returns:
            Tensor: the output fourier embedding
                [..., N_sample, c]
        """
        return torch.cos(
            input=2 * torch.pi * (t_hat_timestep.unsqueeze(dim=-1) * self.w + self.b)
        )


'''
class InputFeatureEmbedder(nn.Module):
    """
    Implements Algorithm 2 in AF3
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
        n_blocks: int = 3,
    ) -> None:
        """
        Args:
            c_atom (int, optional): atom embedding dim. Defaults to 128.
            c_atompair (int, optional): atom pair embedding dim. Defaults to 16.
            c_token (int, optional): token embedding dim. Defaults to 384.
        """
        super(InputFeatureEmbedder, self).__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            num_blocks=n_blocks,
            has_coords=False,
            has_s=False,
            has_z=False,
        )
        # Line2
        # self.input_feature = {"restype": 32} # {"restype": 32, "profile": 32, "deletion_mean": 1}

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): dict of input features
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tensor: token embedding
                [..., N_token, 384 (c_token) + 32 (c_atompair) := 416]
        """
        # Embed per-atom features.
        a, _, _, _ = self.atom_attention_encoder(
            input_feature_dict=input_feature_dict,
        )  # [..., N_token, c_token]
        # Concatenate the per-token features.
        # batch_shape = input_feature_dict["restype"].shape[:-1]
        # s_inputs = torch.cat(
        #    [a]
        #    + [
        #        input_feature_dict[name].reshape(*batch_shape, d)
        #        for name, d in self.input_feature.items()
        #    ],
        #    dim=-1,
        # )

        return a
'''
