"""Tests for codec_video_gen.delta_gen module."""

import pytest
import torch

from codec_video_gen.delta_gen import DeltaPredictor, create_delta_predictor


class TestDeltaPredictor:
    def test_output_shape(self) -> None:
        pred = DeltaPredictor(in_channels=4, hidden_channels=32, num_res_blocks=2)
        x = torch.randn(1, 4, 16, 16)
        with torch.no_grad():
            delta = pred(x)
        assert delta.shape == x.shape

    def test_with_timestep(self) -> None:
        pred = DeltaPredictor(in_channels=4, hidden_channels=32, time_embed_dim=64)
        x = torch.randn(1, 4, 16, 16)
        t = torch.tensor([5])
        with torch.no_grad():
            delta = pred(x, t)
        assert delta.shape == x.shape

    def test_initial_delta_small(self) -> None:
        """Output should be near-zero at initialization."""
        pred = DeltaPredictor(in_channels=4, hidden_channels=32, num_res_blocks=2)
        x = torch.randn(1, 4, 16, 16)
        with torch.no_grad():
            delta = pred(x)
        # Initial delta should be small due to output init
        assert delta.abs().mean().item() < 1.0

    def test_gradient_flow(self) -> None:
        pred = DeltaPredictor(in_channels=4, hidden_channels=32, num_res_blocks=2)
        pred.train()
        x = torch.randn(1, 4, 8, 8, requires_grad=True)
        delta = pred(x)
        loss = delta.sum()
        loss.backward()
        assert x.grad is not None

    def test_no_nan(self) -> None:
        pred = DeltaPredictor(in_channels=4, hidden_channels=64, num_res_blocks=4)
        x = torch.randn(2, 4, 16, 16)
        with torch.no_grad():
            delta = pred(x)
        assert not torch.isnan(delta).any()

    def test_float16(self) -> None:
        pred = DeltaPredictor(in_channels=4, hidden_channels=32).half()
        x = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        with torch.no_grad():
            delta = pred(x)
        assert not torch.isnan(delta).any()
        assert delta.dtype == torch.float16

    def test_param_count(self) -> None:
        pred = DeltaPredictor(in_channels=4, hidden_channels=128, num_res_blocks=4)
        count = pred.count_parameters()
        assert count > 0
        assert count < 200_000_000  # Must be under budget


class TestCreateDeltaPredictor:
    def test_within_budget(self) -> None:
        pred = create_delta_predictor(
            hidden_channels=256, num_res_blocks=8, max_params=5_000_000,
        )
        assert pred.count_parameters() <= 5_000_000

    def test_default_creation(self) -> None:
        pred = create_delta_predictor()
        assert pred is not None
        assert pred.count_parameters() < 200_000_000
