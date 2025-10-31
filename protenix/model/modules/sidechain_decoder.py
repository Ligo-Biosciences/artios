from typing import Dict

import torch
from jaxtyping import Float
from torch import Tensor, nn

from protenix.model.modules.atom_attention import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
)
from protenix.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.utils import centre_random_augmentation, expand_at_dim


class SidechainDecoder(nn.Module):
    """A lightweight non-equivariant sidechain decoder for allatom sequence design."""

    def __init__(
        self,
        node_dim,
        c_atom: int = 128,
        c_atompair: int = 16,
        num_blocks: int = 2,
        num_heads: int = 4,
        num_queries: int = 32,
        num_keys: int = 128,
    ) -> None:
        super().__init__()
        self.relpos = RelativePositionEncoding(r_max=32, s_max=2, c_z=32)
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_token=node_dim,
            has_coords=True,
            has_s=True,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=node_dim,
            c_z=32,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        # Single conditioning embedding
        self.embed_restype = LinearNoBias(32, node_dim)
        self.fourier_embedding = FourierEmbedding(c=node_dim)
        self.embed_timestep = LinearNoBias(node_dim, node_dim)
        self.layernorm_s = LayerNorm(2 * node_dim)
        self.proj_s = LinearNoBias(2 * node_dim, node_dim)

        self.layernorm_a = LayerNorm(node_dim)

        # Decoder
        self.atom_attention_decoder = AtomAttentionDecoder(
            c_token=node_dim,
            c_atom=c_atom,
            c_atompair=c_atompair,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_queries=num_queries,
            num_keys=num_keys,
        )

    def c_out(self, timestep) -> Tensor:
        return 1 - timestep

    def forward(
        self,
        input_feature_dict: Dict[str, Tensor],
        r_l: Float[Tensor, "... n_sample n_atom 3"],
        node_emb: Float[Tensor, "... n_token node_dim"],
        timestep: Float[Tensor, "... n_sample"],
        residual_centre_atoms: Float[Tensor, "... n_sample n_atom 3"],
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """All-atom denoiser forward pass."""
        # Grab mask and indices
        atom_mask = input_feature_dict["atom_mask"]
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        n_token = node_emb.shape[-2]

        z = self.relpos(
            asym_id=input_feature_dict["asym_id"],
            residue_index=input_feature_dict["residue_index"],
            entity_id=input_feature_dict["entity_id"],
            sym_id=input_feature_dict["sym_id"],
            token_index=input_feature_dict["token_index"],
        )  # [..., n_token, n_token, c_z]
        z = z.unsqueeze(-4)  # N_sample dim

        # Compute sequence and node embeddings
        seq_emb = self.embed_restype(input_feature_dict["restype"])
        s = torch.cat([seq_emb, node_emb], dim=-1)
        s = self.proj_s(s)  # [..., n_token, node_dim]

        # Add timestep embedding
        timestep_emb = self.embed_timestep(
            self.fourier_embedding(t_hat_timestep=2 * timestep - 1)
        ).to(z.dtype)  # [..., n_sample, node_dim]
        s = s.unsqueeze(-3) + expand_at_dim(timestep_emb, dim=-2, n=n_token)

        # Encode atoms
        a, q_l, c_l, p_lm, trunked_pair_mask = self.atom_attention_encoder(
            input_feature_dict=input_feature_dict,
            r_l=r_l,
            s=s,
            z=z,
            return_trunked_pair_mask=True,
        )
        a = self.layernorm_a(a)

        # Decode atoms
        # TODO: the atom attention encoder and decoder have different signatures.
        # This is resulting in a lot of shape wrangling which I don't like.
        r_update = self.atom_attention_decoder(
            a=a,
            q_skip=q_l,
            c_skip=c_l,
            p_skip=p_lm,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
            trunked_pair_mask=trunked_pair_mask.squeeze(-4),
        )  # [..., n_sample, n_atom, 3]

        denoised_atoms = residual_centre_atoms + r_update * self.c_out(
            timestep[..., None, None]
        )

        return denoised_atoms

    def sample(
        self,
        input_feature_dict: Dict[str, Tensor],
        node_emb: Float[Tensor, "... n_token node_dim"],
        timesteps: Float[Tensor, "... n_sample"],
        residual_centre_atoms: Float[Tensor, "... n_sample n_atom 3"],
        stoch_type: str = "deterministic",
        alpha: float = 1.0,
        gamma: float = 1.0,
        n_sample: int = 1,
        sigma_data: float = 4.0,
        eps: float = 1e-8,
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """All-atom denoising sampling using diffusion process.

        Performs iterative denoising of atom coordinates through a diffusion sampling
        process with configurable stochasticity schedules and integration methods.

        Args:
            input_feature_dict (Dict[str, Tensor]): input feature dict
            node_emb: Node embeddings from MPNN
            timesteps: Diffusion timesteps for sampling
            residual_centre_atoms: Residue center atom coordinates for residual connection
            stoch_type (str, optional): Type of stochasticity schedule.
                Options include "deterministic", "constant", "non_singular", "zero_ends".
            alpha: Controls the level of noise added during sampling. Defaults to 1.0.
            gamma: Scaling factor for stochastic noise term. Defaults to 1.0.
            n_sample: Number of samples to generate. Defaults to 1.

        Returns:
            Float[Tensor, "... n_sample n_atom 3"]: Denoised atom coordinates
        """
        N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
        batch_shape = input_feature_dict["residue_index"].shape[:-1]
        device = input_feature_dict["residue_index"].device
        dtype = torch.float32  # input_feature_dict["residue_index"].dtype

        def _calculate_velocity_and_score(x_noisy, t_cur):
            """Calculate velocity and scaled score for the diffusion process for a single timestep."""
            coords_pred = self.forward(
                input_feature_dict=input_feature_dict,
                r_l=x_noisy,
                node_emb=node_emb,
                timestep=t_cur,  # [..., n_sample]
                residual_centre_atoms=residual_centre_atoms,
            )

            # Calculate velocity
            t_cur_expanded = t_cur[..., None, None]  # [..., n_sample, 1, 1]
            velocity_pred = (coords_pred - x_noisy) / torch.clamp(
                1 - t_cur_expanded, min=eps
            )

            # Predict score (division by 1-t absorbed into the stochasticity schedule for numerics)
            scaled_score = t_cur_expanded * velocity_pred - x_noisy
            return velocity_pred, scaled_score

        # Stochasticity schedule g (scaled by 1/(1-t) for numerical stability)
        # We also normalise by the functions' integral over [0,1].
        # This allows for the parameter alpha to fully control the level of noise added.
        # Now, the area under the noise schedule is equal to alpha.
        stoch_schedules = {
            "deterministic": lambda t: torch.zeros_like(t),
            "constant": lambda t: torch.ones_like(t)
            * (0.5 * (alpha**2))
            / torch.clamp(1 - t, min=eps),
            "non_singular": lambda t: torch.where(
                t > 0.95,  # do not noise close to 1 for ligand's sake
                torch.zeros_like(t),
                4 * torch.ones_like(t) * 0.5 * (alpha**2),
            ),  # int_{0}^{1}(non_sing) = 1/4
            "zero_ends": lambda t: 12
            * 0.5
            * (alpha**2)
            * t,  # int_{0}^{1}(zero_ends) = 1/12
        }

        if stoch_type not in stoch_schedules:
            raise ValueError(f"Invalid stochasticity type: {stoch_type}")

        g = stoch_schedules[stoch_type](timesteps)

        # Generate starting noise
        x_0 = sigma_data * torch.randn(
            size=(*batch_shape, n_sample, N_atom, 3), device=device, dtype=dtype
        )  # [..., N_sample, N_atom, 3]

        # Initialise Euler/Euler-Maruyama
        x = x_0
        noise = torch.zeros_like(x)

        # Sampling loop
        for i, (t_cur, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # Shape wrangling for current timestep
            t = t_cur.new_full((batch_shape + (n_sample,)), t_cur).to(
                dtype
            )  # [..., N_sample]

            # Generate noise (for stochastic sampling)
            if stoch_type != "deterministic":
                noise = torch.randn_like(x)

            # Stochastic schedule at current time step.
            g_t = g[i]

            # Centre and rotate coords [..., N_sample, N_atom, 3]
            x = (
                centre_random_augmentation(x_input_coords=x, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Calculate velocity and scaled score
            velocity_pred, scaled_score = _calculate_velocity_and_score(
                x_noisy=x,
                t_cur=t,
            )

            # Calculate update
            step = t_next - t_cur
            update = step * (velocity_pred.to(torch.float64) + g_t * scaled_score)

            # Euler step
            x = x + update + torch.sqrt(2 * step * g_t * gamma) * noise

        return x
