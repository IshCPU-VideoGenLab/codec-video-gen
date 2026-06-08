#!/usr/bin/env python
"""Analyze temporal redundancy in video latents.

Usage:
    python scripts/analyze_redundancy.py --num-frames 16 --motion-scale 0.05
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from codec_video_gen.cli import main

if __name__ == "__main__":
    sys.exit(main(["analyze"] + sys.argv[1:]))
