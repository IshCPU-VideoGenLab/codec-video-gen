"""End-to-end codec-style video generation pipeline.

Orchestrates keyframe generation (I-frames) and delta prediction (P-frames)
according to the GOP schedule to produce a complete video sequence.
"""

import gc
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from codec_video_gen.config import CodecConfig
from codec_video_gen.frame_types import FrameType, VideoFrame, VideoSequence
from codec_video_gen.scheduler import GOPScheduler
from codec_video_gen.delta_gen import DeltaPredictor, create_delta_predictor

logger = logging.getLogger(__name__)


class CodecPipeline:
    """Codec-style video generation pipeline.

    Generates videos using a GOP structure: expensive keyframe generation
    for I-frames and lightweight delta prediction for P-frames.

    Two modes of operation:
    1. **With full model**: Uses KeyframeGenerator for I-frames (realistic)
    2. **Standalone**: Uses synthetic/provided keyframes (for testing)

    Args:
        config: Codec pipeline configuration.
    """

    def __init__(self, config: CodecConfig) -> None:
        self._config = config
        self._scheduler = GOPScheduler.from_config(config)
        self._delta_predictor: Optional[DeltaPredictor] = None

        self._setup_delta_predictor()

    def _setup_delta_predictor(self) -> None:
        """Initialize the delta predictor network."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self._config.dtype, torch.float16)

        self._delta_predictor = create_delta_predictor(
            in_channels=self._config.latent_shape[0],
            hidden_channels=self._config.delta_config.hidden_channels,
            num_res_blocks=self._config.delta_config.num_res_blocks,
            time_embed_dim=self._config.delta_config.time_embed_dim,
            max_params=200_000_000,
            dtype=dtype,
        )
        self._delta_predictor.eval()

    @property
    def scheduler(self) -> GOPScheduler:
        """The GOP scheduler."""
        return self._scheduler

    @property
    def delta_predictor(self) -> Optional[DeltaPredictor]:
        """The delta predictor network."""
        return self._delta_predictor

    def generate_p_frame(
        self,
        prev_latent: torch.Tensor,
        frame_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Generate a P-frame by predicting delta from the previous frame.

        Args:
            prev_latent: Previous frame latent, shape (C, H, W).
            frame_index: Index of the frame being generated.

        Returns:
            Tuple of (frame_latent, delta, generation_time_ms).
        """
        if self._delta_predictor is None:
            raise RuntimeError("Delta predictor not initialized")

        # Add batch dimension
        prev = prev_latent.unsqueeze(0)
        t = torch.tensor([frame_index], dtype=torch.long)

        start = time.perf_counter()

        with torch.no_grad():
            delta = self._delta_predictor(prev, t)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Apply delta
        frame_latent = prev + delta
        delta_squeezed = delta.squeeze(0)
        frame_squeezed = frame_latent.squeeze(0)

        return frame_squeezed, delta_squeezed, elapsed_ms

    def generate_from_keyframes(
        self,
        keyframes: Dict[int, torch.Tensor],
    ) -> VideoSequence:
        """Generate a complete video sequence from provided keyframes.

        Takes pre-generated keyframes for I-frame positions and fills in
        P-frames using the delta predictor.

        Args:
            keyframes: Dictionary mapping frame index → latent tensor.
                       Must include all I-frame indices from the schedule.

        Returns:
            Complete VideoSequence with all frames.
        """
        sequence = VideoSequence()
        total_start = time.perf_counter()

        prev_latent = None

        for i in range(self._config.num_frames):
            frame_type = self._scheduler.get_frame_type(i)

            if frame_type == FrameType.I_FRAME:
                # Use provided keyframe
                if i not in keyframes:
                    raise ValueError(
                        f"Keyframe for I-frame at index {i} not provided. "
                        f"Required indices: "
                        f"{[j for j in range(self._config.num_frames) if self._scheduler.get_frame_type(j) == FrameType.I_FRAME]}"
                    )

                latent = keyframes[i]
                frame = VideoFrame(
                    index=i,
                    frame_type=FrameType.I_FRAME,
                    latent=latent,
                    generation_time_ms=0.0,
                )
                prev_latent = latent

            else:
                # Generate P-frame from delta prediction
                if prev_latent is None:
                    raise RuntimeError(
                        f"P-frame at index {i} has no previous frame. "
                        f"First frame must be an I-frame."
                    )

                latent, delta, elapsed = self.generate_p_frame(prev_latent, i)
                delta_norm = torch.norm(delta.float()).item()

                # Quality guard: if delta is too large, flag it
                if delta_norm > self._config.max_delta_norm:
                    logger.warning(
                        "Frame %d: delta norm %.2f exceeds threshold %.2f. "
                        "Consider smaller keyframe interval.",
                        i, delta_norm, self._config.max_delta_norm,
                    )

                frame = VideoFrame(
                    index=i,
                    frame_type=FrameType.P_FRAME,
                    latent=latent,
                    delta=delta,
                    reference_index=i - 1,
                    generation_time_ms=elapsed,
                    delta_norm=delta_norm,
                )
                prev_latent = latent

            sequence.frames.append(frame)

            if self._config.verbose and i % 4 == 0:
                logger.info(
                    "Frame %d/%d [%s] generated",
                    i + 1, self._config.num_frames, frame_type.value,
                )

        total_elapsed = (time.perf_counter() - total_start) * 1000
        sequence.total_generation_time_ms = total_elapsed
        sequence.gops = self._scheduler.gops

        logger.info(
            "Video generated: %d frames (%d I, %d P) in %.1f ms",
            sequence.num_frames, sequence.num_i_frames,
            sequence.num_p_frames, total_elapsed,
        )

        return sequence

    def generate_synthetic(
        self,
        motion_scale: float = 0.05,
        seed: int = 42,
    ) -> VideoSequence:
        """Generate a video using synthetic keyframes (for testing).

        Creates random keyframes and fills P-frames with delta prediction.

        Args:
            motion_scale: Scale of synthetic frame variation.
            seed: Random seed.

        Returns:
            Complete VideoSequence.
        """
        from codec_video_gen.temporal_compress import generate_synthetic_video_latents

        c, h, w = self._config.latent_shape

        # Generate synthetic base frames for keyframe positions
        torch.manual_seed(seed)
        keyframes = {}
        prev_kf = torch.randn(c, h, w, dtype=torch.float16)

        for i in range(self._config.num_frames):
            if self._scheduler.get_frame_type(i) == FrameType.I_FRAME:
                if i == 0:
                    keyframes[i] = prev_kf
                else:
                    # Each keyframe drifts from the last
                    drift = torch.randn(c, h, w, dtype=torch.float16) * motion_scale * 5
                    prev_kf = prev_kf + drift
                    keyframes[i] = prev_kf

        return self.generate_from_keyframes(keyframes)


def compare_codec_vs_baseline(
    config: CodecConfig,
    motion_scale: float = 0.05,
    seed: int = 42,
) -> Dict[str, object]:
    """Compare codec-style generation vs all-I-frame baseline.

    Args:
        config: Codec configuration.
        motion_scale: Synthetic motion scale.
        seed: Random seed.

    Returns:
        Comparison dictionary with timing and quality metrics.
    """
    from codec_video_gen.temporal_compress import (
        generate_synthetic_video_latents,
        analyze_temporal_redundancy,
    )

    # Generate codec-style
    pipeline = CodecPipeline(config)
    codec_result = pipeline.generate_synthetic(motion_scale, seed)

    # Baseline: all I-frames (simulated as same cost per frame)
    baseline_config = CodecConfig(
        keyframe_interval=1,
        num_frames=config.num_frames,
        latent_shape=config.latent_shape,
        output_dir=config.output_dir,
    )
    baseline_pipeline = CodecPipeline(baseline_config)
    baseline_result = baseline_pipeline.generate_synthetic(motion_scale, seed)

    # Compute savings
    codec_savings = pipeline.scheduler.estimate_compute_savings()

    # Analyze redundancy of codec output
    codec_latents = codec_result.get_latent_stack()
    if codec_latents is not None:
        redundancy = analyze_temporal_redundancy(codec_latents)
    else:
        redundancy = {}

    return {
        "codec": codec_result.summary(),
        "baseline": baseline_result.summary(),
        "savings": codec_savings,
        "redundancy": redundancy.get("aggregate", {}),
        "config": {
            "keyframe_interval": config.keyframe_interval,
            "num_frames": config.num_frames,
            "motion_scale": motion_scale,
        },
    }
