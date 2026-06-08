"""Frame type abstractions for codec-inspired video generation.

Defines the I-frame (keyframe) and P-frame (predicted/delta) types,
along with utilities for working with sequences of typed frames.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class FrameType(Enum):
    """Type of frame in the GOP structure."""

    I_FRAME = "I"  # Intra-coded: generated from scratch
    P_FRAME = "P"  # Predicted: generated as delta from previous frame


@dataclass
class VideoFrame:
    """A single frame in the codec-style generation pipeline.

    Args:
        index: Frame index in the sequence (0-based).
        frame_type: Whether this is an I-frame or P-frame.
        latent: The latent tensor for this frame, shape (C, H, W).
        delta: The predicted delta tensor (P-frames only).
        reference_index: Index of the reference frame (P-frames only).
        generation_time_ms: Wall-clock time to generate this frame.
        delta_norm: L2 norm of the delta (P-frames only).
    """

    index: int
    frame_type: FrameType
    latent: Optional[torch.Tensor] = None
    delta: Optional[torch.Tensor] = None
    reference_index: Optional[int] = None
    generation_time_ms: float = 0.0
    delta_norm: float = 0.0

    @property
    def is_keyframe(self) -> bool:
        """Whether this frame is an I-frame (keyframe)."""
        return self.frame_type == FrameType.I_FRAME

    @property
    def is_predicted(self) -> bool:
        """Whether this frame is a P-frame (predicted from delta)."""
        return self.frame_type == FrameType.P_FRAME


@dataclass
class GOPInfo:
    """Information about a Group of Pictures.

    Args:
        start_index: Index of the first frame (I-frame) in this GOP.
        end_index: Index of the last frame in this GOP (exclusive).
        keyframe_interval: Number of frames in this GOP.
        num_i_frames: Number of I-frames (always 1).
        num_p_frames: Number of P-frames.
    """

    start_index: int
    end_index: int
    keyframe_interval: int
    num_i_frames: int = 1
    num_p_frames: int = 0

    def __post_init__(self) -> None:
        self.num_p_frames = (self.end_index - self.start_index) - self.num_i_frames

    @property
    def total_frames(self) -> int:
        return self.end_index - self.start_index

    @property
    def compression_ratio(self) -> float:
        """Theoretical compression ratio (P-frames are ~10x cheaper)."""
        if self.total_frames == 0:
            return 1.0
        p_frame_cost = 0.1  # P-frame costs ~10% of I-frame
        total_cost = self.num_i_frames + self.num_p_frames * p_frame_cost
        baseline_cost = self.total_frames  # All I-frames
        return baseline_cost / total_cost


@dataclass
class VideoSequence:
    """A complete sequence of typed frames.

    Args:
        frames: List of VideoFrame objects in order.
        gops: List of GOPInfo describing the GOP structure.
        total_generation_time_ms: Total time to generate all frames.
    """

    frames: List[VideoFrame] = field(default_factory=list)
    gops: List[GOPInfo] = field(default_factory=list)
    total_generation_time_ms: float = 0.0

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def num_i_frames(self) -> int:
        return sum(1 for f in self.frames if f.is_keyframe)

    @property
    def num_p_frames(self) -> int:
        return sum(1 for f in self.frames if f.is_predicted)

    @property
    def avg_delta_norm(self) -> float:
        """Average delta norm across P-frames."""
        norms = [f.delta_norm for f in self.frames if f.is_predicted and f.delta_norm > 0]
        return sum(norms) / len(norms) if norms else 0.0

    def get_latent_stack(self) -> Optional[torch.Tensor]:
        """Stack all frame latents into a single tensor.

        Returns:
            Tensor of shape (num_frames, C, H, W), or None if frames lack latents.
        """
        latents = [f.latent for f in self.frames if f.latent is not None]
        if not latents:
            return None
        return torch.stack(latents, dim=0)

    def summary(self) -> Dict[str, object]:
        """Generate a summary dictionary."""
        return {
            "num_frames": self.num_frames,
            "num_i_frames": self.num_i_frames,
            "num_p_frames": self.num_p_frames,
            "num_gops": len(self.gops),
            "avg_delta_norm": round(self.avg_delta_norm, 4),
            "total_time_ms": round(self.total_generation_time_ms, 1),
            "avg_time_per_frame_ms": round(
                self.total_generation_time_ms / max(self.num_frames, 1), 1
            ),
        }
