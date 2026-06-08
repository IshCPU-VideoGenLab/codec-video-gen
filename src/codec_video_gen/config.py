"""Configuration for codec-video-gen."""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DeltaPredictorConfig:
    """Configuration for the delta predictor network.

    Args:
        channels: Number of latent channels (must match VAE output).
        hidden_channels: Hidden channel dimension in residual blocks.
        num_res_blocks: Number of residual blocks.
        time_embed_dim: Dimension of timestep embedding.
        use_attention: Whether to use a single attention layer (expensive).
        dropout: Dropout rate.
    """

    channels: int = 4
    hidden_channels: int = 128
    num_res_blocks: int = 4
    time_embed_dim: int = 128
    use_attention: bool = False
    dropout: float = 0.0


@dataclass
class CodecConfig:
    """Configuration for codec-style video generation.

    Args:
        model_name: Base model name (e.g., "Wan-AI/Wan2.1-T2V-1.3B-Diffusers").
        model_path: Local path to model weights.
        keyframe_interval: Number of frames between I-frames (GOP size).
        num_frames: Total number of frames to generate.
        output_dir: Directory for results.
        delta_model_scale: Relative size of delta predictor vs full model.
        delta_config: Delta predictor network configuration.
        error_correction: Apply correction at GOP boundaries to prevent drift.
        max_delta_norm: Maximum allowed delta norm before forcing an I-frame.
        dtype: Data type for model loading.
        low_memory: Enable memory-efficient mode.
        latent_shape: Shape of a single latent frame (C, H, W).
        verbose: Print progress.
    """

    model_name: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    model_path: Optional[str] = None
    keyframe_interval: int = 8
    num_frames: int = 16
    output_dir: str = "results"
    delta_model_scale: float = 0.1
    delta_config: DeltaPredictorConfig = field(default_factory=DeltaPredictorConfig)
    error_correction: bool = True
    max_delta_norm: float = 10.0
    dtype: str = "float16"
    low_memory: bool = True
    latent_shape: Tuple[int, int, int] = (4, 32, 32)
    verbose: bool = True

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        if self.keyframe_interval < 1:
            raise ValueError("keyframe_interval must be at least 1")
        if self.num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        if self.delta_model_scale <= 0 or self.delta_model_scale > 1:
            raise ValueError("delta_model_scale must be in (0, 1]")


@dataclass
class RedundancyConfig:
    """Configuration for temporal redundancy analysis.

    Args:
        model_name: Model to use for generating latent frames.
        model_path: Local path to model weights.
        num_frames: Number of frames to analyze.
        num_samples: Number of different video samples to average over.
        resolution: Input resolution (H, W).
        output_dir: Directory for results.
        dtype: Data type.
    """

    model_name: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    model_path: Optional[str] = None
    num_frames: int = 16
    num_samples: int = 5
    resolution: Tuple[int, int] = (256, 256)
    output_dir: str = "results"
    dtype: str = "float16"

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
