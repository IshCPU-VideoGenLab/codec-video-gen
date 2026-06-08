"""Command-line interface for codec-video-gen."""

import argparse
import logging
import sys
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="codec-video-gen",
        description="Codec-inspired temporal design for CPU-native video generation.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- generate ---
    gen = sub.add_parser("generate", help="Generate video with codec pipeline")
    gen.add_argument("--model", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    gen.add_argument("--num-frames", type=int, default=16)
    gen.add_argument("--keyframe-interval", type=int, default=8)
    gen.add_argument("--output", type=str, default="results")
    gen.add_argument("--motion-scale", type=float, default=0.05)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--debug", action="store_true")

    # --- analyze ---
    ana = sub.add_parser("analyze", help="Analyze temporal redundancy")
    ana.add_argument("--num-frames", type=int, default=16)
    ana.add_argument("--motion-scale", type=float, default=0.05)
    ana.add_argument("--output", type=str, default="results")
    ana.add_argument("--debug", action="store_true")

    # --- compare ---
    cmp = sub.add_parser("compare", help="Compare codec vs baseline")
    cmp.add_argument("--num-frames", type=int, default=16)
    cmp.add_argument("--keyframe-interval", type=int, default=8)
    cmp.add_argument("--motion-scale", type=float, default=0.05)
    cmp.add_argument("--output", type=str, default="results")
    cmp.add_argument("--debug", action="store_true")

    # --- gop ---
    gop = sub.add_parser("gop", help="Visualize GOP structure")
    gop.add_argument("--num-frames", type=int, default=32)
    gop.add_argument("--keyframe-interval", type=int, default=8)

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    if args.command is None:
        print("Usage: codec-video-gen {generate|analyze|compare|gop} [options]")
        return 1

    level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    if args.command == "generate":
        from codec_video_gen.config import CodecConfig
        from codec_video_gen.pipeline import CodecPipeline
        from codec_video_gen.report import save_json_report

        config = CodecConfig(
            num_frames=args.num_frames,
            keyframe_interval=args.keyframe_interval,
            output_dir=args.output,
        )
        pipeline = CodecPipeline(config)
        result = pipeline.generate_synthetic(args.motion_scale, args.seed)
        save_json_report(result.summary(), args.output, "generation_results.json")
        print(f"\nGenerated {result.num_frames} frames ({result.num_i_frames} I, {result.num_p_frames} P)")
        return 0

    elif args.command == "analyze":
        from codec_video_gen.temporal_compress import (
            generate_synthetic_video_latents,
            analyze_temporal_redundancy,
        )
        from codec_video_gen.report import save_json_report, format_redundancy_summary

        frames = generate_synthetic_video_latents(
            num_frames=args.num_frames, motion_scale=args.motion_scale,
        )
        analysis = analyze_temporal_redundancy(frames)
        save_json_report(analysis, args.output, "redundancy_analysis.json")
        print(format_redundancy_summary(analysis))
        return 0

    elif args.command == "compare":
        from codec_video_gen.config import CodecConfig
        from codec_video_gen.pipeline import compare_codec_vs_baseline
        from codec_video_gen.report import save_json_report, format_comparison_summary

        config = CodecConfig(
            num_frames=args.num_frames,
            keyframe_interval=args.keyframe_interval,
            output_dir=args.output,
        )
        comparison = compare_codec_vs_baseline(config, args.motion_scale)
        save_json_report(comparison, args.output, "codec_comparison.json")
        print(format_comparison_summary(comparison))
        return 0

    elif args.command == "gop":
        from codec_video_gen.scheduler import GOPScheduler
        from codec_video_gen.report import format_gop_visual

        scheduler = GOPScheduler(args.keyframe_interval, args.num_frames)
        print(format_gop_visual(scheduler.format_visual()))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
