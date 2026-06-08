"""Tests for codec_video_gen.scheduler module."""

import pytest

from codec_video_gen.frame_types import FrameType
from codec_video_gen.scheduler import GOPScheduler


class TestGOPScheduler:
    def test_first_frame_always_i(self) -> None:
        scheduler = GOPScheduler(keyframe_interval=8, num_frames=16)
        assert scheduler.get_frame_type(0) == FrameType.I_FRAME

    def test_keyframe_at_intervals(self) -> None:
        scheduler = GOPScheduler(keyframe_interval=4, num_frames=16)
        for i in range(16):
            expected = FrameType.I_FRAME if i % 4 == 0 else FrameType.P_FRAME
            assert scheduler.get_frame_type(i) == expected, f"Frame {i} wrong"

    def test_counts(self) -> None:
        scheduler = GOPScheduler(keyframe_interval=8, num_frames=16)
        assert scheduler.num_i_frames == 2
        assert scheduler.num_p_frames == 14

    def test_all_i_frames(self) -> None:
        """Interval=1 means every frame is an I-frame."""
        scheduler = GOPScheduler(keyframe_interval=1, num_frames=8)
        assert scheduler.num_i_frames == 8
        assert scheduler.num_p_frames == 0

    def test_single_frame(self) -> None:
        scheduler = GOPScheduler(keyframe_interval=8, num_frames=1)
        assert scheduler.num_i_frames == 1
        assert scheduler.num_p_frames == 0

    def test_reference_index_i_frame(self) -> None:
        scheduler = GOPScheduler(keyframe_interval=8, num_frames=16)
        assert scheduler.get_reference_index(0) == 0  # Self-reference

    def test_reference_index_p_frame(self) -> None:
        scheduler = GOPScheduler(keyframe_interval=8, num_frames=16)
        assert scheduler.get_reference_index(3) == 2  # Previous frame

    def test_invalid_index(self) -> None:
        scheduler = GOPScheduler(keyframe_interval=8, num_frames=16)
        with pytest.raises(IndexError):
            scheduler.get_frame_type(20)

    def test_invalid_interval(self) -> None:
        with pytest.raises(ValueError):
            GOPScheduler(keyframe_interval=0, num_frames=16)

    def test_gops_structure(self) -> None:
        scheduler = GOPScheduler(keyframe_interval=4, num_frames=12)
        gops = scheduler.gops
        assert len(gops) == 3
        assert gops[0].start_index == 0
        assert gops[0].end_index == 4
        assert gops[1].start_index == 4
        assert gops[2].start_index == 8

    def test_compute_savings(self) -> None:
        scheduler = GOPScheduler(keyframe_interval=8, num_frames=16)
        savings = scheduler.estimate_compute_savings(p_frame_cost=0.1)
        assert savings["speedup"] > 1.0
        assert savings["savings_pct"] > 0

    def test_format_visual(self) -> None:
        scheduler = GOPScheduler(keyframe_interval=4, num_frames=8)
        visual = scheduler.format_visual()
        assert "I" in visual
        assert "P" in visual
