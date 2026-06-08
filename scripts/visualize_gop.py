#!/usr/bin/env python
"""Visualize GOP structure.

Usage:
    python scripts/visualize_gop.py --num-frames 32 --keyframe-interval 8
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from codec_video_gen.cli import main

if __name__ == "__main__":
    sys.exit(main(["gop"] + sys.argv[1:]))
