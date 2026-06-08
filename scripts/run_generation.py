#!/usr/bin/env python
"""Run codec-style video generation.

Usage:
    python scripts/run_generation.py --num-frames 16 --keyframe-interval 8
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from codec_video_gen.cli import main

if __name__ == "__main__":
    sys.exit(main(["generate"] + sys.argv[1:]))
