"""Report generation for codec-video-gen results."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


def save_json_report(data: Dict[str, Any], output_dir: str, filename: str) -> str:
    """Save a report as JSON.

    Args:
        data: Report data.
        output_dir: Output directory.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Report saved to: %s", path)
    return path


def format_redundancy_summary(analysis: Dict[str, Any]) -> str:
    """Format temporal redundancy analysis as readable text.

    Args:
        analysis: Output from analyze_temporal_redundancy().

    Returns:
        Formatted string.
    """
    agg = analysis.get("aggregate", {})
    lines = [
        "",
        "=" * 60,
        "  Temporal Redundancy Analysis",
        "=" * 60,
        "",
        f"  Frames analyzed:         {analysis.get('num_frames', '?')}",
        f"  Frame pairs:             {analysis.get('num_pairs', '?')}",
        "",
        f"  Avg redundancy:          {agg.get('avg_redundancy_pct', '?')}%",
        f"  Avg cosine similarity:   {agg.get('avg_cosine_similarity', '?')}",
        f"  Avg relative change:     {agg.get('avg_relative_change', '?')}",
        f"  Avg MSE:                 {agg.get('avg_mse', '?')}",
        "",
        "=" * 60,
        "",
    ]
    return "\n".join(lines)


def format_comparison_summary(comparison: Dict[str, Any]) -> str:
    """Format codec vs baseline comparison as readable text.

    Args:
        comparison: Output from compare_codec_vs_baseline().

    Returns:
        Formatted string.
    """
    codec = comparison.get("codec", {})
    baseline = comparison.get("baseline", {})
    savings = comparison.get("savings", {})

    lines = [
        "",
        "=" * 60,
        "  Codec vs Baseline Comparison",
        "=" * 60,
        "",
        f"  {'Metric':<30} {'Baseline':>12} {'Codec':>12}",
        "  " + "-" * 56,
        f"  {'Total time (ms)':<30} {baseline.get('total_time_ms', 0):>12.1f} {codec.get('total_time_ms', 0):>12.1f}",
        f"  {'Avg time/frame (ms)':<30} {baseline.get('avg_time_per_frame_ms', 0):>12.1f} {codec.get('avg_time_per_frame_ms', 0):>12.1f}",
        f"  {'I-frames':<30} {baseline.get('num_i_frames', 0):>12} {codec.get('num_i_frames', 0):>12}",
        f"  {'P-frames':<30} {baseline.get('num_p_frames', 0):>12} {codec.get('num_p_frames', 0):>12}",
        "  " + "-" * 56,
        "",
        f"  Theoretical speedup:     {savings.get('speedup', '?')}x",
        f"  Compute savings:         {savings.get('savings_pct', '?')}%",
        "",
        "=" * 60,
        "",
    ]
    return "\n".join(lines)


def format_gop_visual(scheduler_visual: str) -> str:
    """Wrap scheduler visual output for display.

    Args:
        scheduler_visual: Output from GOPScheduler.format_visual().

    Returns:
        Formatted string with header.
    """
    lines = [
        "",
        "=" * 60,
        "  GOP Structure",
        "=" * 60,
        "",
        scheduler_visual,
        "",
        "=" * 60,
        "",
    ]
    return "\n".join(lines)
