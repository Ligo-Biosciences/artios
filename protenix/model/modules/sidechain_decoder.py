import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from protenix.data.features import FullContext
from protenix.model.modules.atom_attention import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
)
from protenix.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.modules.transformer import DiffusionTransformer
from protenix.model.utils import centre_random_augmentation
from protenix.utils.typing import Float, typecheck


class SidechainDecoder(nn.Module):
    """A lightweight non-equivariant sidechain decoder for all-atom sequence design."""

    def __init__(
        self,
        c_s: int = 384,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_z: int = 32,
        atom_encoder: dict[str, int] = {"num_blocks": 3, "num_heads": 4},
        transformer: dict[str, int] = {"num_blocks": 10, "num_heads": 12},
        atom_decoder: dict[str, int] = {"num_blocks": 3, "num_heads": 4},
        num_queries: int = 32,
        num_keys: int = 128,
        sigma_coord: float = 10.0,
    ) -> None:
        super().__init__()
        self.sigma_coord = sigma_coord
        self.relpos = RelativePositionEncoding(r_max=32, s_max=2, c_z=c_z)

        # Encoder
        self.atom_attention_encoder = AtomAttentionEncoder(
            **atom_encoder,
            c_token=c_token,
            has_coords=True,
            has_s=True,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            c_z=c_z,
            num_queries=num_queries,
            num_keys=num_keys,
        )

        # Single conditioning embedding
        self.embed_restype = LinearNoBias(32, c_s)
        self.fourier_embedding = FourierEmbedding(c=c_s)
        self.embed_timestep = LinearNoBias(c_s, c_s)

        # DiT
        self.diffusion_transformer = DiffusionTransformer(
            **transformer,
            c_token=c_token,
            c_s=c_s,
            c_z=c_z,
        )
        self.layernorm_a = LayerNorm(c_token)

        # Decoder
        self.atom_attention_decoder = AtomAttentionDecoder(
            **atom_decoder,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            num_queries=num_queries,
            num_keys=num_keys,
        )

    def c_out(self, timestep: Tensor) -> Tensor:
        return 1 - timestep

    @typecheck
    def forward(
        self,
        context: FullContext,
        x_noisy: Float[Tensor, "... n_sample n_atom 3"],
        timestep: Float[Tensor, "... n_sample"],
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """All-atom denoiser forward pass.

        Args:
            context: Full feature context.
            x_noisy: Noisy atom coordinates.
            timestep: Diffusion timestep.

        Returns:
            Denoised atom coordinates.
        """
        # Extract features from context
        dtype = context.reference.ref_pos.dtype
        atom_mask = context.mask.atom_mask
        atom_to_token_idx = context.index.atom_to_token_idx
        token_mask = context.mask.token_mask
        restype = context.token.restype
        n_token = restype.shape[-2]

        # Compute pair mask
        token_mask = token_mask.unsqueeze(-2)  # add n_sample dim
        mask_i = rearrange(token_mask, "... n -> ... n 1")
        mask_j = rearrange(token_mask, "... n -> ... 1 n")
        pair_mask = mask_i & mask_j

        # Compute relative position encoding
        z = self.relpos(
            asym_id=context.token.asym_id,
            residue_index=context.token.residue_index,
            entity_id=context.token.entity_id,
            sym_id=context.token.sym_id,
            token_index=context.token.token_index,
        )
        z = rearrange(z, "... i j c -> ... 1 i j c")

        # Compute sequence embedding
        s = self.embed_restype(restype.to(dtype))

        # Add timestep embedding
        timestep_emb = self.embed_timestep(
            self.fourier_embedding(t_hat_timestep=2 * timestep - 1)
        ).to(z.dtype)
        s = rearrange(s, "... n c -> ... 1 n c")
        s = s + repeat(timestep_emb, "... n_sample c -> ... n_sample n c", n=n_token)

        # Encode atoms
        a, skip = self.atom_attention_encoder(
            context=context,
            r_l=x_noisy / self.sigma_coord,
            s=s,
            z=z,
        )

        a = self.diffusion_transformer(a=a, s=s, z=z, pair_mask=pair_mask)
        a = self.layernorm_a(a)

        # Decode atoms
        r_update = self.atom_attention_decoder(
            a=a,
            skip=skip,
            atom_to_token_idx=atom_to_token_idx,
            atom_mask=atom_mask,
        )
        denoised_atoms = x_noisy + r_update * self.c_out(timestep[..., None, None])
        return denoised_atoms

    @typecheck
    def sample(
        self,
        context: FullContext,
        timesteps: Float[Tensor, "n_steps"],
        residual_centre_atoms: Float[Tensor, "... n_sample n_atom 3"],
        stoch_type: str = "deterministic",
        alpha: float = 1.0,
        gamma: float = 1.0,
        n_sample: int = 1,
        sigma_data: float = 1.0,
        eps: float = 1e-5,
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """All-atom denoising sampling using diffusion process.

        Args:
            context: Full feature context.
            timesteps: Diffusion timesteps for sampling.
            residual_centre_atoms: Residue center atom coordinates for residual connection.
            stoch_type: Type of stochasticity schedule.
                Options: "deterministic", "constant", "non_singular", "zero_ends".
            alpha: Controls the level of noise added during sampling.
            gamma: Scaling factor for stochastic noise term.
            n_sample: Number of samples to generate.
            sigma_data: Standard deviation for initial noise.
            eps: Small constant for numerical stability.

        Returns:
            Denoised atom coordinates.
        """
        n_atom = context.index.atom_to_token_idx.shape[-1]
        batch_shape = context.token.residue_index.shape[:-1]
        device = context.token.residue_index.device
        dtype = torch.float32

        def _calculate_velocity_and_score(
            x_noisy: Tensor, t_cur: Tensor
        ) -> tuple[Tensor, Tensor]:
            """Calculate velocity and scaled score for a single timestep."""
            coords_pred = self.forward(
                context=context,
                x_noisy=x_noisy,
                timestep=t_cur,
            )

            t_cur_ = repeat(t_cur, "... -> ... 1 1")
            velocity_pred = (coords_pred - x_noisy) / torch.clamp(1 - t_cur_, min=eps)
            scaled_score = t_cur_ * velocity_pred - x_noisy
            return velocity_pred, scaled_score

        # Stochasticity schedules
        stoch_schedules = {
            "deterministic": lambda t: torch.zeros_like(t),
            "constant": lambda t: torch.ones_like(t)
            * (0.5 * (alpha**2))
            / torch.clamp(1 - t, min=eps),
            "non_singular": lambda t: torch.where(
                t > 0.95,
                torch.zeros_like(t),
                4 * torch.ones_like(t) * 0.5 * (alpha**2),
            ),
            "zero_ends": lambda t: 12 * 0.5 * (alpha**2) * t,
        }

        if stoch_type not in stoch_schedules:
            raise ValueError(f"Invalid stochasticity type: {stoch_type}")

        g = stoch_schedules[stoch_type](timesteps)

        # Generate starting noise
        x_0 = sigma_data * torch.randn(
            size=(*batch_shape, n_sample, n_atom, 3), device=device, dtype=dtype
        )
        x_0 = x_0 + residual_centre_atoms

        # Initialise Euler/Euler-Maruyama
        x = x_0
        noise = torch.zeros_like(x)

        # Sampling loop
        for i, (t_cur, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t = t_cur.new_full((*batch_shape, n_sample), t_cur).to(dtype)

            if stoch_type != "deterministic":
                noise = torch.randn_like(x)

            g_t = g[i]

            # Centre and rotate coords
            x = (
                centre_random_augmentation(x_input_coords=x, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            velocity_pred, scaled_score = _calculate_velocity_and_score(
                x_noisy=x, t_cur=t
            )

            step = t_next - t_cur
            update = step * (velocity_pred.to(torch.float64) + g_t * scaled_score)
            x = x + update + torch.sqrt(2 * step * g_t * gamma) * noise

        return x
