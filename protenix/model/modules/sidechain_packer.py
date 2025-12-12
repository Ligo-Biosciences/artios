import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from protenix.data.features import FullContext
from protenix.model.generator import (
    Deterministic,
    FlowMatchingSampler,
    InferenceTimestepScheduler,
    StochasticitySchedule,
    VelocityScorePredictor,
)
from protenix.model.modules.atom_attention import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
)
from protenix.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from protenix.model.modules.primitives import LayerNorm, LinearNoBias
from protenix.model.modules.transformer import DiffusionTransformer
from protenix.utils.typing import Float, typecheck


class SidechainPacker(nn.Module):
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
        local_sigma_coord: float = 1.0,
    ) -> None:
        super().__init__()
        self.sigma_coord = sigma_coord
        self.local_sigma_coord = local_sigma_coord
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

    def as_predictor(
        self,
        context: FullContext,
        eps: float = 1e-5,
    ) -> VelocityScorePredictor:
        """Create a VelocityScorePredictor bound to this packer and context.

        Args:
            context: Full feature context to use for predictions.
            eps: Small constant for numerical stability.

        Returns:
            A callable satisfying VelocityScorePredictor protocol.
        """

        def predictor(
            x_noisy: Float[Tensor, "... n_atom 3"],
            timestep: Float[Tensor, "..."],
        ) -> tuple[Float[Tensor, "... n_atom 3"], Float[Tensor, "... n_atom 3"]]:
            coords_pred = self.forward(
                context=context,
                r_noisy=x_noisy,
                timestep=timestep,
            )

            t = repeat(timestep, "... -> ... 1 1")
            velocity = (coords_pred - x_noisy) / torch.clamp(1 - t, min=eps)
            score = t * velocity - x_noisy
            return velocity, score

        return predictor

    @typecheck
    def sample(
        self,
        context: FullContext,
        timestep_scheduler: InferenceTimestepScheduler,
        n_steps: int,
        residual_centre_atoms: Float[Tensor, "... n_sample n_atom 3"],
        stochasticity: StochasticitySchedule = Deterministic(),
        gamma: float = 1.0,
        n_sample: int = 1,
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """All-atom denoising sampling using flow matching.

        Args:
            context: Full feature context.
            timestep_scheduler: Scheduler for generating timestep sequence.
            n_steps: Number of integration steps.
            residual_centre_atoms: Residue center atom coordinates for residual connection.
            stochasticity: Schedule defining g(t) in the SDE.
            gamma: Temperature parameter for the stochastic term.
            n_sample: Number of samples to generate.

        Returns:
            Denoised atom coordinates.
        """
        n_atom = context.index.atom_to_token_idx.shape[-1]
        batch_shape = context.token.residue_index.shape[:-1]
        device = context.token.residue_index.device
        dtype = torch.float32

        # Generate initial noise with residual offset
        x_init = self.local_sigma_coord * torch.randn(
            size=(*batch_shape, n_sample, n_atom, 3), device=device, dtype=dtype
        )
        x_init = x_init + residual_centre_atoms

        # Create sampler and predictor
        sampler = FlowMatchingSampler(
            timestep_scheduler=timestep_scheduler,
            stochasticity=stochasticity,
            gamma=gamma,
            second_order=False,
            centre_coords=True,
        )
        predictor = self.as_predictor(context)

        return sampler.sample(
            predictor=predictor,
            x_init=x_init,
            n_steps=n_steps,
        )

    @typecheck
    def f_forward(
        self,
        r_noisy: Float[Tensor, "... n_sample n_atom 3"],
        timestep: Float[Tensor, "... n_sample"],
        context: FullContext,
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """All-atom denoiser forward pass.

        Args:
            r_noisy: Noisy atom coordinates.
            timestep: Diffusion timestep.
            context: Full feature context.

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
            r_l=r_noisy,  # / self.sigma_coord,
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
        # denoised_atoms = r_noisy + r_update * self.c_out(timestep[..., None, None])
        return r_update

    def c_out(self, timestep: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """Output scaling factor (velocity parametrization)."""
        return 1 - timestep

    @typecheck
    def forward(
        self,
        r_noisy: Float[Tensor, "... n_sample n_atom 3"],
        timestep: Float[Tensor, "... n_sample"],
        context: FullContext,
    ) -> Float[Tensor, "... n_sample n_atom 3"]:
        """Performs a forward pass of the sidechain packer.

        Args:
            r_noisy: Noisy atom coordinates.
            timestep: Diffusion timestep.
            context: Full context containing all features and labels.
            inplace_safe: Whether it is safe to use inplace operations.

        Returns:
            Denoised atom coordinates.
        """
        # Expand timesteps
        timestep_ = timestep[..., None, None]

        # Scale input coords to unit variance (no timestep dependence)
        r_noisy_unit = r_noisy / self.sigma_coord

        # Predict velocity
        out = self.f_forward(
            r_noisy=r_noisy_unit,
            timestep=timestep,
            context=context,
        )

        # Apply skip connection and output scaling
        r_pred = r_noisy + out * self.c_out(timestep_)

        return r_pred
