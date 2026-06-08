"""Temporal redundancy analysis for video latent sequences.

Measures how much information is shared between consecutive frames
to validate the codec approach. High redundancy = codec works well.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_frame_similarity(
    frame_a: torch.Tensor,
    frame_b: torch.Tensor,
) -> Dict[str, float]:
    """Compute similarity metrics between two latent frames.

    Args:
        frame_a: First frame latent, shape (C, H, W).
        frame_b: Second frame latent, shape (C, H, W).

    Returns:
        Dictionary with MSE, cosine similarity, and L1 distance.
    """
    a = frame_a.float().flatten()
    b = frame_b.float().flatten()

    mse = ((a - b) ** 2).mean().item()
    l1 = (a - b).abs().mean().item()

    dot = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    cosine = (dot / (norm_a * norm_b + 1e-8)).item()

    # Relative change: how much of the signal is delta vs original
    signal_norm = norm_a.item()
    delta_norm = torch.norm(a - b).item()
    relative_change = delta_norm / (signal_norm + 1e-8)

    return {
        "mse": mse,
        "l1": l1,
        "cosine_similarity": cosine,
        "delta_norm": delta_norm,
        "signal_norm": signal_norm,
        "relative_change": relative_change,
        "redundancy_pct": max(0, (1 - relative_change)) * 100,
    }


def analyze_temporal_redundancy(
    frames: torch.Tensor,
) -> Dict[str, object]:
    """Analyze temporal redundancy across a sequence of latent frames.

    Computes frame-to-frame similarity for all consecutive pairs
    and aggregates the results.

    Args:
        frames: Tensor of latent frames, shape (T, C, H, W).

    Returns:
        Dictionary with per-pair metrics and aggregate statistics.
    """
    num_frames = frames.shape[0]
    if num_frames < 2:
        logger.warning("Need at least 2 frames for redundancy analysis")
        return {"error": "insufficient_frames"}

    pair_metrics = []

    for t in range(num_frames - 1):
        metrics = compute_frame_similarity(frames[t], frames[t + 1])
        metrics["frame_pair"] = f"{t}->{t+1}"
        pair_metrics.append(metrics)

    # Aggregate statistics
    redundancies = [m["redundancy_pct"] for m in pair_metrics]
    cosines = [m["cosine_similarity"] for m in pair_metrics]
    mses = [m["mse"] for m in pair_metrics]
    relative_changes = [m["relative_change"] for m in pair_metrics]

    return {
        "num_frames": num_frames,
        "num_pairs": len(pair_metrics),
        "per_pair": pair_metrics,
        "aggregate": {
            "avg_redundancy_pct": round(np.mean(redundancies), 2),
            "min_redundancy_pct": round(np.min(redundancies), 2),
            "max_redundancy_pct": round(np.max(redundancies), 2),
            "avg_cosine_similarity": round(np.mean(cosines), 4),
            "avg_mse": round(np.mean(mses), 6),
            "avg_relative_change": round(np.mean(relative_changes), 4),
        },
    }


def analyze_redundancy_vs_distance(
    frames: torch.Tensor,
    max_distance: int = 16,
) -> List[Dict[str, float]]:
    """Analyze how redundancy changes with frame distance.

    Computes similarity between frames separated by distance d,
    for d = 1, 2, ..., max_distance.

    Args:
        frames: Latent frames, shape (T, C, H, W).
        max_distance: Maximum frame distance to analyze.

    Returns:
        List of dicts with distance and average similarity metrics.
    """
    num_frames = frames.shape[0]
    results = []

    for d in range(1, min(max_distance + 1, num_frames)):
        cosines = []
        relative_changes = []

        for t in range(num_frames - d):
            metrics = compute_frame_similarity(frames[t], frames[t + d])
            cosines.append(metrics["cosine_similarity"])
            relative_changes.append(metrics["relative_change"])

        results.append({
            "distance": d,
            "avg_cosine_similarity": round(np.mean(cosines), 4),
            "avg_relative_change": round(np.mean(relative_changes), 4),
            "avg_redundancy_pct": round((1 - np.mean(relative_changes)) * 100, 2),
            "num_pairs": len(cosines),
        })

    return results


def compute_optimal_keyframe_interval(
    frames: torch.Tensor,
    quality_threshold: float = 0.8,
) -> int:
    """Estimate the optimal keyframe interval based on redundancy.

    Finds the largest frame distance where average cosine similarity
    stays above the threshold. Beyond this distance, P-frames would
    be too different from their reference to predict well.

    Args:
        frames: Latent frames, shape (T, C, H, W).
        quality_threshold: Minimum cosine similarity for P-frame viability.

    Returns:
        Recommended keyframe interval.
    """
    distance_analysis = analyze_redundancy_vs_distance(frames)

    optimal = 1
    for entry in distance_analysis:
        if entry["avg_cosine_similarity"] >= quality_threshold:
            optimal = entry["distance"]
        else:
            break

    logger.info(
        "Optimal keyframe interval: %d (threshold=%.2f)",
        optimal, quality_threshold,
    )
    return optimal


def generate_synthetic_video_latents(
    num_frames: int = 16,
    channels: int = 4,
    height: int = 32,
    width: int = 32,
    motion_scale: float = 0.05,
    seed: int = 42,
) -> torch.Tensor:
    """Generate synthetic video latents with controlled temporal redundancy.

    Creates a sequence of latent frames where each frame is a small
    perturbation of the previous one. Useful for testing the codec
    pipeline without a real generative model.

    Args:
        num_frames: Number of frames to generate.
        channels: Latent channels.
        height: Latent height.
        width: Latent width.
        motion_scale: Scale of frame-to-frame changes (smaller = more redundant).
        seed: Random seed.

    Returns:
        Tensor of shape (num_frames, channels, height, width).
    """
    torch.manual_seed(seed)

    frames = torch.zeros(num_frames, channels, height, width)
    frames[0] = torch.randn(channels, height, width)

    for t in range(1, num_frames):
        delta = torch.randn(channels, height, width) * motion_scale
        frames[t] = frames[t - 1] + delta

    logger.info(
        "Generated %d synthetic latent frames (motion_scale=%.3f)",
        num_frames, motion_scale,
    )
    return frames
