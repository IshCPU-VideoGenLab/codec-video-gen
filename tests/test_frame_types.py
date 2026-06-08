"""Tests for codec_video_gen.frame_types module."""

import pytest
import torch

from codec_video_gen.frame_types import FrameType, VideoFrame, VideoSequence, GOPInfo


class TestFrameType:
    def test_i_frame_value(self) -> None:
        assert FrameType.I_FRAME.value == "I"

    def test_p_frame_value(self) -> None:
        assert FrameType.P_FRAME.value == "P"


class TestVideoFrame:
    def test_i_frame_properties(self) -> None:
        frame = VideoFrame(index=0, frame_type=FrameType.I_FRAME)
        assert frame.is_keyframe is True
        assert frame.is_predicted is False

    def test_p_frame_properties(self) -> None:
        frame = VideoFrame(index=1, frame_type=FrameType.P_FRAME, reference_index=0)
        assert frame.is_keyframe is False
        assert frame.is_predicted is True

    def test_with_latent(self) -> None:
        latent = torch.randn(4, 32, 32)
        frame = VideoFrame(index=0, frame_type=FrameType.I_FRAME, latent=latent)
        assert frame.latent is not None
        assert frame.latent.shape == (4, 32, 32)


class TestGOPInfo:
    def test_basic(self) -> None:
        gop = GOPInfo(start_index=0, end_index=8, keyframe_interval=8)
        assert gop.total_frames == 8
        assert gop.num_i_frames == 1
        assert gop.num_p_frames == 7

    def test_compression_ratio(self) -> None:
        gop = GOPInfo(start_index=0, end_index=8, keyframe_interval=8)
        # 1 I + 7 P at 0.1 cost = 1.7 vs baseline 8
        assert gop.compression_ratio == pytest.approx(8 / 1.7, rel=0.01)


class TestVideoSequence:
    def test_counts(self) -> None:
        seq = VideoSequence(frames=[
            VideoFrame(0, FrameType.I_FRAME),
            VideoFrame(1, FrameType.P_FRAME),
            VideoFrame(2, FrameType.P_FRAME),
            VideoFrame(3, FrameType.P_FRAME),
        ])
        assert seq.num_frames == 4
        assert seq.num_i_frames == 1
        assert seq.num_p_frames == 3

    def test_latent_stack(self) -> None:
        latents = [torch.randn(4, 8, 8) for _ in range(3)]
        seq = VideoSequence(frames=[
            VideoFrame(0, FrameType.I_FRAME, latent=latents[0]),
            VideoFrame(1, FrameType.P_FRAME, latent=latents[1]),
            VideoFrame(2, FrameType.P_FRAME, latent=latents[2]),
        ])
        stack = seq.get_latent_stack()
        assert stack is not None
        assert stack.shape == (3, 4, 8, 8)

    def test_summary(self) -> None:
        seq = VideoSequence(frames=[
            VideoFrame(0, FrameType.I_FRAME),
        ], total_generation_time_ms=100.0)
        s = seq.summary()
        assert s["num_frames"] == 1
        assert s["total_time_ms"] == 100.0
