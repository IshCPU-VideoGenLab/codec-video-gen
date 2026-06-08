"""GOP (Group of Pictures) scheduler for codec-style video generation.

Determines which frames are keyframes (I-frames) and which are predicted
frames (P-frames) based on the keyframe interval configuration.
"""

import logging
from typing import Dict, List, Tuple

from codec_video_gen.config import CodecConfig
from codec_video_gen.frame_types import FrameType, GOPInfo

logger = logging.getLogger(__name__)


class GOPScheduler:
    """Schedules frame types according to a GOP structure.

    Assigns I-frame or P-frame type to each frame index based on
    the keyframe interval. The first frame is always an I-frame.

    Args:
        keyframe_interval: Number of frames between keyframes.
        num_frames: Total number of frames in the video.
    """

    def __init__(self, keyframe_interval: int, num_frames: int) -> None:
        if keyframe_interval < 1:
            raise ValueError("keyframe_interval must be at least 1")
        if num_frames < 1:
            raise ValueError("num_frames must be at least 1")

        self._interval = keyframe_interval
        self._num_frames = num_frames
        self._schedule = self._build_schedule()
        self._gops = self._build_gops()

    @classmethod
    def from_config(cls, config: CodecConfig) -> "GOPScheduler":
        """Create a scheduler from a CodecConfig.

        Args:
            config: Codec configuration.

        Returns:
            A new GOPScheduler instance.
        """
        return cls(
            keyframe_interval=config.keyframe_interval,
            num_frames=config.num_frames,
        )

    def _build_schedule(self) -> List[FrameType]:
        """Build the frame type schedule.

        Returns:
            List of FrameType for each frame index.
        """
        schedule = []
        for i in range(self._num_frames):
            if i % self._interval == 0:
                schedule.append(FrameType.I_FRAME)
            else:
                schedule.append(FrameType.P_FRAME)
        return schedule

    def _build_gops(self) -> List[GOPInfo]:
        """Build GOP information from the schedule.

        Returns:
            List of GOPInfo for each group of pictures.
        """
        gops = []
        gop_start = 0

        for i in range(1, self._num_frames):
            if self._schedule[i] == FrameType.I_FRAME:
                gops.append(GOPInfo(
                    start_index=gop_start,
                    end_index=i,
                    keyframe_interval=i - gop_start,
                ))
                gop_start = i

        # Final GOP
        gops.append(GOPInfo(
            start_index=gop_start,
            end_index=self._num_frames,
            keyframe_interval=self._num_frames - gop_start,
        ))

        return gops

    def get_frame_type(self, index: int) -> FrameType:
        """Get the frame type for a given index.

        Args:
            index: Frame index (0-based).

        Returns:
            FrameType for this frame.

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= self._num_frames:
            raise IndexError(
                f"Frame index {index} out of range [0, {self._num_frames})"
            )
        return self._schedule[index]

    def get_reference_index(self, index: int) -> int:
        """Get the reference frame index for a P-frame.

        For P-frames, returns the index of the previous frame.
        For I-frames, returns the frame's own index (self-reference).

        Args:
            index: Frame index.

        Returns:
            Reference frame index.
        """
        if self._schedule[index] == FrameType.I_FRAME:
            return index
        return index - 1

    @property
    def schedule(self) -> List[FrameType]:
        """The complete frame type schedule."""
        return list(self._schedule)

    @property
    def gops(self) -> List[GOPInfo]:
        """List of GOP information."""
        return list(self._gops)

    @property
    def num_i_frames(self) -> int:
        """Total number of I-frames."""
        return sum(1 for ft in self._schedule if ft == FrameType.I_FRAME)

    @property
    def num_p_frames(self) -> int:
        """Total number of P-frames."""
        return sum(1 for ft in self._schedule if ft == FrameType.P_FRAME)

    def estimate_compute_savings(
        self,
        i_frame_cost: float = 1.0,
        p_frame_cost: float = 0.1,
    ) -> Dict[str, float]:
        """Estimate compute savings from the codec approach.

        Args:
            i_frame_cost: Relative cost of generating one I-frame (default: 1.0).
            p_frame_cost: Relative cost of one P-frame as fraction of I-frame.

        Returns:
            Dictionary with baseline cost, codec cost, savings, and speedup.
        """
        baseline = self._num_frames * i_frame_cost
        codec = self.num_i_frames * i_frame_cost + self.num_p_frames * p_frame_cost

        savings_pct = (1 - codec / baseline) * 100 if baseline > 0 else 0
        speedup = baseline / codec if codec > 0 else 0

        return {
            "baseline_cost": baseline,
            "codec_cost": round(codec, 2),
            "savings_pct": round(savings_pct, 1),
            "speedup": round(speedup, 2),
            "num_i_frames": self.num_i_frames,
            "num_p_frames": self.num_p_frames,
        }

    def format_visual(self) -> str:
        """Generate a visual representation of the GOP structure.

        Returns:
            Multi-line string showing the frame schedule.
        """
        lines = []

        # Frame indices
        indices = " ".join(f"{i:>3}" for i in range(self._num_frames))
        lines.append(f"Frame: {indices}")

        # Frame types
        types = " ".join(f"  {ft.value}" for ft in self._schedule)
        lines.append(f"Type:  {types}")

        # Cost visualization
        cost_chars = []
        for ft in self._schedule:
            if ft == FrameType.I_FRAME:
                cost_chars.append("███")
            else:
                cost_chars.append("  ▓")
        costs = " ".join(cost_chars)
        lines.append(f"Cost:  {costs}")

        # Summary
        savings = self.estimate_compute_savings()
        lines.append("")
        lines.append(
            f"I-frames: {self.num_i_frames}, "
            f"P-frames: {self.num_p_frames}, "
            f"Speedup: {savings['speedup']}x, "
            f"Savings: {savings['savings_pct']}%"
        )

        return "\n".join(lines)
