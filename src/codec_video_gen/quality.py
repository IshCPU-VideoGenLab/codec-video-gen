"""Quality evaluation for codec-style video generation.

Measures quality differences between codec-generated and baseline-generated
frames, with special attention to error drift across P-frames within a GOP.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def frame_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """MSE between two frames."""
    return ((a.float() - b.float()) ** 2).mean().item()


def frame_psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    """PSNR between two frames."""
    mse = frame_mse(a, b)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


def frame_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two frames."""
    af = a.float().flatten()
    bf = b.float().flatten()
    dot = torch.dot(af, bf)
    na = torch.norm(af)
    nb = torch.norm(bf)
    return (dot / (na * nb + 1e-8)).item()


def evaluate_frame_quality(
    codec_frames: torch.Tensor,
    reference_frames: torch.Tensor,
) -> List[Dict[str, float]]:
    """Evaluate per-frame quality between codec and reference outputs.

    Args:
        codec_frames: Codec-generated frames, shape (T, C, H, W).
        reference_frames: Reference (all-I-frame) frames, shape (T, C, H, W).

    Returns:
        List of per-frame quality metrics.
    """
    num_frames = min(codec_frames.shape[0], reference_frames.shape[0])
    results = []

    for t in range(num_frames):
        metrics = {
            "frame": t,
            "mse": frame_mse(codec_frames[t], reference_frames[t]),
            "psnr_db": frame_psnr(codec_frames[t], reference_frames[t]),
            "cosine_similarity": frame_cosine(codec_frames[t], reference_frames[t]),
        }
        results.append(metrics)

    return results


def analyze_error_drift(
    codec_frames: torch.Tensor,
    reference_frames: torch.Tensor,
    frame_types: List[str],
) -> Dict[str, object]:
    """Analyze how quality degrades across P-frames within GOPs.

    Error accumulates as each P-frame is predicted from the previous one.
    This function measures whether quality drops steadily within a GOP
    and recovers at I-frame boundaries.

    Args:
        codec_frames: Codec-generated frames.
        reference_frames: Reference frames.
        frame_types: List of "I" or "P" for each frame.

    Returns:
        Analysis with per-frame errors and drift statistics.
    """
    per_frame = evaluate_frame_quality(codec_frames, reference_frames)

    # Track position within GOP
    p_frame_position = 0
    position_errors: Dict[int, List[float]] = {}

    for i, metrics in enumerate(per_frame):
        ft = frame_types[i] if i < len(frame_types) else "P"

        if ft == "I":
            p_frame_position = 0
        else:
            p_frame_position += 1

        metrics["gop_position"] = p_frame_position
        metrics["frame_type"] = ft

        if ft == "P":
            if p_frame_position not in position_errors:
                position_errors[p_frame_position] = []
            position_errors[p_frame_position].append(metrics["mse"])

    # Compute average error by position within GOP
    drift_by_position = []
    for pos in sorted(position_errors.keys()):
        errors = position_errors[pos]
        drift_by_position.append({
            "position": pos,
            "avg_mse": round(np.mean(errors), 6),
            "max_mse": round(np.max(errors), 6),
            "num_samples": len(errors),
        })

    # Check if error increases with position (drift)
    if len(drift_by_position) >= 2:
        first_pos_mse = drift_by_position[0]["avg_mse"]
        last_pos_mse = drift_by_position[-1]["avg_mse"]
        drift_ratio = last_pos_mse / (first_pos_mse + 1e-10)
    else:
        drift_ratio = 1.0

    return {
        "per_frame": per_frame,
        "drift_by_position": drift_by_position,
        "drift_ratio": round(drift_ratio, 4),
        "has_significant_drift": drift_ratio > 2.0,
    }


def compute_gop_quality_summary(
    codec_frames: torch.Tensor,
    reference_frames: torch.Tensor,
    keyframe_interval: int,
) -> Dict[str, object]:
    """Compute quality summary aggregated by GOP.

    Args:
        codec_frames: Codec-generated frames.
        reference_frames: Reference frames.
        keyframe_interval: Keyframe interval used.

    Returns:
        Quality summary with per-GOP and aggregate metrics.
    """
    num_frames = min(codec_frames.shape[0], reference_frames.shape[0])

    # Assign frame types
    frame_types = []
    for i in range(num_frames):
        if i % keyframe_interval == 0:
            frame_types.append("I")
        else:
            frame_types.append("P")

    # Per-frame quality
    per_frame = evaluate_frame_quality(codec_frames, reference_frames)

    # Aggregate
    i_frame_mses = [m["mse"] for m, ft in zip(per_frame, frame_types) if ft == "I"]
    p_frame_mses = [m["mse"] for m, ft in zip(per_frame, frame_types) if ft == "P"]
    all_cosines = [m["cosine_similarity"] for m in per_frame]

    return {
        "keyframe_interval": keyframe_interval,
        "num_frames": num_frames,
        "avg_mse_all": round(np.mean([m["mse"] for m in per_frame]), 6),
        "avg_mse_i_frames": round(np.mean(i_frame_mses), 6) if i_frame_mses else 0,
        "avg_mse_p_frames": round(np.mean(p_frame_mses), 6) if p_frame_mses else 0,
        "avg_cosine_all": round(np.mean(all_cosines), 4),
        "avg_psnr_all": round(np.mean([m["psnr_db"] for m in per_frame]), 2),
    }
