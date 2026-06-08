# CLAUDE.md — codec-video-gen

> This file is read by Claude Code at the start of every session.
> It provides full context so you never have to re-explain the project.

---

## Project Identity

- **Org:** IshCPU-VideoGenLab
- **Repo:** codec-video-gen
- **Author:** Ishmael Affum Kwakye (Calyx)
- **GitHub:** calyxish
- **Institution:** University of Ghana, Legon
- **Phase:** 3 of 7

---

## What This Project Is

This is the **most novel and most publishable** phase of the entire project.

We redesign the video generation loop to work like a video codec (H.264/H.265).
Instead of generating every frame from scratch (as all current video diffusion
models do), we generate **keyframes** (I-frames) at full quality and then
**predict deltas** (P-frames) between consecutive frames.

This eliminates the ~95% temporal redundancy between consecutive video frames,
slashing compute by an order of magnitude.

**codec-video-gen** implements:

1. **Frame type classification** — I-frames (keyframes) vs P-frames (deltas)
2. **GOP (Group of Pictures) scheduling** — which frames are I vs P
3. **Keyframe generation** — full diffusion generation for I-frames
4. **Delta prediction** — lightweight network that predicts frame-to-frame changes
5. **Temporal compression analysis** — measuring actual redundancy in video latents
6. **Full pipeline** — end-to-end codec-inspired video generation

---

## The Core Insight — Why This Works

### How H.264 Encodes Video on CPUs in Real Time

H.264 doesn't compress every frame independently. It classifies frames:

- **I-frame (Intra)**: Encoded from scratch. Full quality. Expensive.
- **P-frame (Predicted)**: Encoded as a delta from the previous frame.
  Only the motion/changes are stored. 10-50× cheaper than I-frames.
- **B-frame (Bidirectional)**: Predicted from both past and future frames.
  Even cheaper but adds latency. We skip these for simplicity.

A typical GOP structure: `I P P P P P P I P P P P P P I ...`
One keyframe every 8-16 frames. Everything else is deltas.

### Applying This to Video Generation

Current video diffusion models generate ALL frames through the full denoising
pipeline. Every frame gets the same expensive treatment regardless of how
similar it is to the previous frame.

Our approach:
1. Generate keyframes (I-frames) through the full model
2. For P-frames, run a lightweight delta predictor that takes:
   - The previous frame's latent
   - A timestep/noise signal
   - Optional text conditioning
   And produces only the CHANGE from the previous frame.

The delta predictor is a tiny fraction of the full model's compute because
frame-to-frame changes are typically small (a person moves slightly,
background stays the same).

---

## Phase 1 → Phase 2 → Phase 3 Handoff

- **Phase 1** (wan-profiler): Told us WHERE compute goes in the model
- **Phase 2** (mamba-video): Replaced O(n²) attention with O(n) SSM
- **Phase 3** (this repo): Reduces HOW MANY times we run the model

Phase 2 made each forward pass cheaper. Phase 3 reduces the number of
expensive forward passes needed.

**Combined effect**: fewer passes × cheaper passes = multiplicative speedup.

---

## The Bigger Picture

| Phase | Repo | Status |
|-------|------|--------|
| 1 | wan-profiler | ✅ Complete |
| 2 | mamba-video | ✅ Complete |
| **3** | **codec-video-gen** (this repo) | **Active** |
| 4 | bitnet-video | Planned |
| 5 | simd-kernels | Planned |
| 6 | (distributed) | Planned |
| 7 | cpu-video-gen | Planned |

---

## Hardware

- **Primary (development + benchmarking):** MacBook Air M4 — ARM64 / NEON, no GPU.
- **Supported, CI-verified:** commodity x86 with AVX2 (any modern Intel/AMD CPU).
- **Origin, proof-of-concept (retired):** Intel Pentium Gold 7505 — x86-64 / AVX2, 2C/4T, 16 GB.

CPU-native, no GPU, across **both** architectures. We develop on the M4, but **all code must stay
within the commodity-hardware design budget** — assume **2–4 cores, 16 GB RAM (~12 GB usable),
no GPU**, and it must run on x86 (AVX2) **and** ARM (NEON). The Pentium Gold proved the
weakest-hardware case. Python 3.9, venv (no conda).

### What This Means For Code:

- **CPU-only.** No CUDA anywhere.
- **The delta predictor must be TINY.** If the full model is 1.3B params,
  the delta predictor should be 50-200M params maximum. Smaller = better.
- **Memory budget: ~12 GB usable.** Full model + delta predictor must fit.
- **float16 everywhere.**
- **GOP analysis on small videos only** (8-32 frames, 256×256).

---

## Code Conventions

- **Python 3.9** — `List[str]` not `list[str]`, no `match` statements
- **Type hints** on all function signatures
- **Docstrings** Google style on all public functions
- **Logging** via `logging`, never `print()` for production
- **Tests** in `tests/` using `pytest`
- **Results** as JSON

---

## File Structure

```
codec-video-gen/
├── CLAUDE.md                ← You are here
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── lessons.md
├── tasks/
│   └── todo.md
├── .claude/
│   ├── settings.json
│   ├── commands/
│   │   ├── review.md
│   │   └── progress.md
│   └── rules/
│       └── python.md
├── configs/
│   └── default.json
├── src/
│   └── codec_video_gen/
│       ├── __init__.py
│       ├── config.py            ← Configuration
│       ├── frame_types.py       ← I-frame / P-frame abstractions
│       ├── keyframe_gen.py      ← Full generation for keyframes
│       ├── delta_gen.py         ← Lightweight delta predictor
│       ├── scheduler.py         ← GOP structure and frame scheduling
│       ├── pipeline.py          ← End-to-end generation pipeline
│       ├── temporal_compress.py ← Redundancy analysis tools
│       ├── quality.py           ← Quality metrics
│       ├── report.py            ← Report generation
│       └── cli.py               ← CLI entry point
├── scripts/
│   ├── run_generation.py
│   ├── analyze_redundancy.py
│   └── visualize_gop.py
├── tests/
│   ├── __init__.py
│   ├── test_frame_types.py
│   ├── test_scheduler.py
│   ├── test_pipeline.py
│   └── test_delta_gen.py
├── results/
│   └── .gitkeep
└── docs/
    └── codec_design.md
```

---

## Key Terminology

- **GOP (Group of Pictures)**: A sequence of frames from one I-frame to
  the next. Typical size: 8-16 frames.
- **I-frame**: Intra-coded frame. Generated independently, full quality.
- **P-frame**: Predicted frame. Generated as delta from previous frame.
- **Delta/Residual**: The difference between consecutive frames in latent space.
- **Temporal redundancy**: The fraction of information shared between
  consecutive frames (~90-98% for typical video).
- **Keyframe interval**: Number of frames between I-frames.

---

## Research Questions This Phase Answers

1. **How much temporal redundancy exists** in Wan 1.3B's latent frames?
2. **How small can the delta predictor be** while maintaining quality?
3. **What's the optimal keyframe interval** (GOP size)?
4. **What's the total compute savings** from codec-inspired generation?
5. **Does quality degrade gracefully** as GOP size increases?

---

## Task Management

Check `tasks/todo.md` before starting any work session.

## Lessons Learned

Check `lessons.md` before writing new code.
