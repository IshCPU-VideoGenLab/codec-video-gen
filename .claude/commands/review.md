# /project:review

1. Read `CLAUDE.md` and `lessons.md` first
2. Review all Python files in `src/codec_video_gen/` for:
   - Type hints, Google-style docstrings, `logging` (no `print`)
   - Python 3.9 compatibility
   - CPU-only (no CUDA), memory safety (16GB budget)
   - Delta predictor size (must be ≤ 200M params)
   - GOP logic correctness (I-frames at interval boundaries)
3. Run `pytest tests/ -v`
4. Report: **Critical** / **Important** / **Suggestions**
