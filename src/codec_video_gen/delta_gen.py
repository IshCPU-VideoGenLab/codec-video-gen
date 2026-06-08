"""Lightweight delta predictor for P-frame generation.

Predicts the change (delta) between consecutive video frames in latent
space. Much smaller than the full generative model — targeting 50-100M
parameters vs the full model's 1.3B.

The key insight: frame-to-frame changes are small and local. A compact
ConvNet with residual blocks can predict these deltas without the full
capacity of a diffusion transformer.
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding, same as used in diffusion models.

    Args:
        dim: Embedding dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute timestep embedding.

        Args:
            t: Timestep tensor, shape (batch,).

        Returns:
            Embedding tensor, shape (batch, dim).
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device).float() / half_dim
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return self.mlp(embedding)


class ResidualBlock(nn.Module):
    """Residual convolutional block with optional timestep conditioning.

    Architecture: Conv → GroupNorm → SiLU → Conv → GroupNorm → + residual

    Args:
        channels: Number of input/output channels.
        time_embed_dim: Dimension of timestep embedding (0 to disable).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        channels: int,
        time_embed_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.time_proj = None
        if time_embed_dim > 0:
            self.time_proj = nn.Linear(time_embed_dim, channels)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, channels, H, W).
            time_emb: Optional timestep embedding, shape (batch, time_embed_dim).

        Returns:
            Output tensor, same shape as input.
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add timestep conditioning
        if time_emb is not None and self.time_proj is not None:
            t = self.time_proj(F.silu(time_emb))
            h = h + t.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return x + h


class DeltaPredictor(nn.Module):
    """Lightweight network for predicting frame-to-frame deltas.

    Takes the previous frame's latent representation and predicts the
    change (delta) to produce the next frame. Optionally conditioned
    on a timestep signal.

    Architecture:
        prev_latent → Conv(in) → [ResBlock × N] → Conv(out) → delta
        + optional timestep conditioning via embedding

    The output delta is added to the previous frame to get the next frame:
        next_frame = prev_frame + delta_predictor(prev_frame, t)

    Args:
        in_channels: Number of latent channels (typically 4).
        hidden_channels: Hidden channel dimension.
        num_res_blocks: Number of residual blocks.
        time_embed_dim: Timestep embedding dimension (0 to disable).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 128,
        num_res_blocks: int = 4,
        time_embed_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # Timestep embedding
        self.time_embed = None
        if time_embed_dim > 0:
            self.time_embed = TimestepEmbedding(time_embed_dim)

        # Input convolution: latent channels → hidden channels
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, time_embed_dim, dropout)
            for _ in range(num_res_blocks)
        ])

        # Output convolution: hidden channels → latent channels (delta)
        self.norm_out = nn.GroupNorm(min(8, hidden_channels), hidden_channels)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

        # Initialize output to near-zero so initial delta is small
        self._init_output()

    def _init_output(self) -> None:
        """Initialize output layer to produce near-zero deltas initially."""
        with torch.no_grad():
            self.conv_out.weight.mul_(0.01)
            if self.conv_out.bias is not None:
                self.conv_out.bias.zero_()

    def forward(
        self,
        prev_latent: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the delta from the previous frame to the next frame.

        Args:
            prev_latent: Previous frame's latent, shape (batch, C, H, W).
            timestep: Optional timestep tensor, shape (batch,).

        Returns:
            Predicted delta tensor, same shape as prev_latent.
        """
        # Timestep embedding
        time_emb = None
        if timestep is not None and self.time_embed is not None:
            time_emb = self.time_embed(timestep)

        # Input projection
        h = self.conv_in(prev_latent)

        # Residual blocks
        for block in self.res_blocks:
            h = block(h, time_emb)

        # Output projection
        h = self.norm_out(h)
        h = F.silu(h)
        delta = self.conv_out(h)

        return delta

    def count_parameters(self) -> int:
        """Count total parameters in the delta predictor.

        Returns:
            Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        """String representation."""
        params = self.count_parameters()
        return (
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"params={params:,} ({params / 1e6:.1f}M)"
        )


def create_delta_predictor(
    in_channels: int = 4,
    hidden_channels: int = 128,
    num_res_blocks: int = 4,
    time_embed_dim: int = 128,
    max_params: int = 200_000_000,
    dtype: torch.dtype = torch.float16,
) -> DeltaPredictor:
    """Create a delta predictor within the parameter budget.

    Automatically adjusts hidden_channels if the requested configuration
    would exceed the parameter budget.

    Args:
        in_channels: Latent channels.
        hidden_channels: Initial hidden channel dimension.
        num_res_blocks: Number of residual blocks.
        time_embed_dim: Timestep embedding dimension.
        max_params: Maximum allowed parameters.
        dtype: Data type for parameters.

    Returns:
        A DeltaPredictor instance within budget.
    """
    predictor = DeltaPredictor(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_res_blocks=num_res_blocks,
        time_embed_dim=time_embed_dim,
    )

    param_count = predictor.count_parameters()

    # Shrink if over budget
    while param_count > max_params and hidden_channels > 16:
        hidden_channels = hidden_channels // 2
        logger.info(
            "Delta predictor too large (%s params). "
            "Reducing hidden_channels to %d.",
            f"{param_count:,}", hidden_channels,
        )
        predictor = DeltaPredictor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            time_embed_dim=time_embed_dim,
        )
        param_count = predictor.count_parameters()

    predictor = predictor.to(dtype)

    logger.info(
        "Delta predictor created: %s params (%.1fM), hidden=%d, blocks=%d",
        f"{param_count:,}", param_count / 1e6,
        hidden_channels, num_res_blocks,
    )

    return predictor
