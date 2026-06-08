"""Tests for codec_video_gen.pipeline module."""

import pytest
import torch

from codec_video_gen.config import CodecConfig
from codec_video_gen.frame_types import FrameType
from codec_video_gen.pipeline import CodecPipeline, compare_codec_vs_baseline


class TestCodecPipeline:
    def _make_config(self, **kwargs) -> CodecConfig:
        import tempfile
        defaults = {
            "output_dir": tempfile.mkdtemp(),
            "num_frames": 8,
            "keyframe_interval": 4,
            "latent_shape": (4, 8, 8),
        }
        defaults.update(kwargs)
        return CodecConfig(**defaults)

    def test_generate_synthetic(self) -> None:
        config = self._make_config()
        pipeline = CodecPipeline(config)
        result = pipeline.generate_synthetic()
        assert result.num_frames == 8
        assert result.num_i_frames == 2  # Frames 0, 4
        assert result.num_p_frames == 6

    def test_all_frames_have_latents(self) -> None:
        config = self._make_config(num_frames=4, keyframe_interval=2)
        pipeline = CodecPipeline(config)
        result = pipeline.generate_synthetic()
        for frame in result.frames:
            assert frame.latent is not None

    def test_p_frames_have_deltas(self) -> None:
        config = self._make_config(num_frames=4, keyframe_interval=4)
        pipeline = CodecPipeline(config)
        result = pipeline.generate_synthetic()
        for frame in result.frames:
            if frame.frame_type == FrameType.P_FRAME:
                assert frame.delta is not None
                assert frame.reference_index is not None

    def test_latent_stack_shape(self) -> None:
        config = self._make_config(num_frames=6, latent_shape=(4, 8, 8))
        pipeline = CodecPipeline(config)
        result = pipeline.generate_synthetic()
        stack = result.get_latent_stack()
        assert stack is not None
        assert stack.shape == (6, 4, 8, 8)

    def test_single_frame(self) -> None:
        config = self._make_config(num_frames=1, keyframe_interval=1)
        pipeline = CodecPipeline(config)
        result = pipeline.generate_synthetic()
        assert result.num_frames == 1
        assert result.num_i_frames == 1

    def test_all_i_frames(self) -> None:
        config = self._make_config(num_frames=4, keyframe_interval=1)
        pipeline = CodecPipeline(config)
        result = pipeline.generate_synthetic()
        assert result.num_i_frames == 4
        assert result.num_p_frames == 0

    def test_generate_from_keyframes(self) -> None:
        config = self._make_config(num_frames=8, keyframe_interval=4, latent_shape=(4, 8, 8))
        pipeline = CodecPipeline(config)

        keyframes = {
            0: torch.randn(4, 8, 8, dtype=torch.float16),
            4: torch.randn(4, 8, 8, dtype=torch.float16),
        }
        result = pipeline.generate_from_keyframes(keyframes)
        assert result.num_frames == 8

    def test_missing_keyframe_raises(self) -> None:
        config = self._make_config(num_frames=8, keyframe_interval=4, latent_shape=(4, 8, 8))
        pipeline = CodecPipeline(config)

        # Only provide keyframe at 0, missing at 4
        keyframes = {0: torch.randn(4, 8, 8, dtype=torch.float16)}
        with pytest.raises(ValueError, match="Keyframe"):
            pipeline.generate_from_keyframes(keyframes)


class TestCompareCodecVsBaseline:
    def test_comparison_runs(self) -> None:
        import tempfile
        config = CodecConfig(
            num_frames=8,
            keyframe_interval=4,
            latent_shape=(4, 8, 8),
            output_dir=tempfile.mkdtemp(),
        )
        result = compare_codec_vs_baseline(config, motion_scale=0.05)
        assert "codec" in result
        assert "baseline" in result
        assert "savings" in result
        assert result["savings"]["speedup"] > 1.0
