<p align="center">
  <img src="https://raw.githubusercontent.com/IshCPU-VideoGenLab/.github/main/logo.svg" alt="IshCPU-VideoGenLab" width="80">
</p>

# codec-video-gen

**Codec-inspired temporal design for CPU-native video generation — generate keyframes, predict deltas.**

Part of [IshCPU-VideoGenLab](https://github.com/IshCPU-VideoGenLab) — building the first video generation model that trains and runs entirely on commodity CPUs.

---

## The Idea

Every video diffusion model generates all frames through the same expensive pipeline. But consecutive video frames are ~95% identical. This is a massive waste of compute.

**codec-video-gen** borrows from how H.264 compresses video in real time on CPUs:

- **I-frames (keyframes)**: Generated through the full model. Full quality. Expensive.
- **P-frames (predicted)**: Generated as lightweight deltas from the previous frame. 10-50× cheaper.

A 16-frame video with keyframe interval 8 needs only 2 full generations + 14 cheap delta predictions, instead of 16 full generations.

---

## The Bigger Picture

This is **Phase 3** of a 7-phase research project:

| Phase | Repo | Goal |
|-------|------|------|
| 1 | [wan-profiler](https://github.com/IshCPU-VideoGenLab/wan-profiler) | Profile compute distribution |
| 2 | [mamba-video](https://github.com/IshCPU-VideoGenLab/mamba-video) | Replace attention with Mamba/SSM |
| **3** | **codec-video-gen** (this repo) | **Codec-inspired temporal design** |
| 4 | bitnet-video | 1-bit quantization (BitNet) |
| 5 | simd-kernels | Portable SIMD execution engine (AVX2 + NEON) |
| 6 | (distributed) | Distributed CPU training |
| 7 | cpu-video-gen | Flagship paper repo |

Phase 2 made each forward pass cheaper (O(n²) → O(n)). Phase 3 reduces how many expensive passes are needed. The effects multiply.

---

## Features

- **GOP scheduler** — configurable I-frame/P-frame patterns with variable keyframe intervals
- **Temporal redundancy analyzer** — measures actual inter-frame similarity in latent space
- **Delta predictor** — lightweight network (~50-200M params) for frame-to-frame prediction
- **Full generation pipeline** — end-to-end codec-style video generation
- **Quality evaluation** — compare codec-style vs standard generation
- **Compute savings calculator** — exact FLOPs and time reduction

---

## Installation

```bash
git clone https://github.com/IshCPU-VideoGenLab/codec-video-gen.git
cd codec-video-gen

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### 1. Analyze Temporal Redundancy

```bash
# Measure how much frames share in latent space
python scripts/analyze_redundancy.py --model wan-1.3b --num-frames 16 --output results/
```

### 2. Run Codec-Style Generation

```bash
# Generate video with I-frame + P-frame structure
python scripts/run_generation.py \
    --model wan-1.3b \
    --num-frames 16 \
    --keyframe-interval 8 \
    --output results/

# Compare with standard (all-I-frame) generation
python scripts/run_generation.py \
    --model wan-1.3b \
    --num-frames 16 \
    --keyframe-interval 1 \
    --output results/baseline/
```

### 3. Visualize GOP Structure

```bash
python scripts/visualize_gop.py --num-frames 32 --keyframe-interval 8 --output results/
```

### Python API

```python
from codec_video_gen.config import CodecConfig
from codec_video_gen.scheduler import GOPScheduler
from codec_video_gen.pipeline import CodecPipeline

config = CodecConfig(
    keyframe_interval=8,
    num_frames=16,
    delta_model_scale=0.1,  # Delta predictor is 10% of full model
)

scheduler = GOPScheduler(config)
pipeline = CodecPipeline(config)

# Generate video
frames = pipeline.generate(prompt="a cat walking")
```

---

## How It Works

### GOP Structure

```
Frame:  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
Type:   I   P   P   P   P   P   P   P   I   P   P   P   P   P   P   P
Cost:   ███ ▓   ▓   ▓   ▓   ▓   ▓   ▓   ███ ▓   ▓   ▓   ▓   ▓   ▓   ▓
```

- `███` = Full model forward pass (expensive)
- `▓` = Delta predictor only (cheap)

### The Delta Predictor

```
prev_frame_latent ─┐
                    ├─→ [Conv layers + Residual blocks] → delta
noise / timestep ───┘

next_frame = prev_frame + delta
```

The delta predictor is a small convolutional network (50-200M params) that
learns to predict frame-to-frame changes in latent space.

---

## Project Structure

```
codec-video-gen/
├── src/codec_video_gen/
│   ├── config.py            # Configuration
│   ├── frame_types.py       # I-frame / P-frame abstractions
│   ├── keyframe_gen.py      # Full keyframe generation
│   ├── delta_gen.py         # Lightweight delta predictor network
│   ├── scheduler.py         # GOP scheduling
│   ├── pipeline.py          # End-to-end pipeline
│   ├── temporal_compress.py # Redundancy analysis
│   ├── quality.py           # Quality metrics
│   ├── report.py            # Report generation
│   └── cli.py               # CLI
├── scripts/                 # Convenience scripts
├── tests/                   # Unit tests
├── configs/                 # Default configurations
└── docs/codec_design.md     # Full methodology
```

---

## Citation

```bibtex
@software{kwakye2026codecvideogen,
  author = {Kwakye, Ishmael Affum},
  title = {codec-video-gen: Codec-Inspired Temporal Design for CPU-Native Video Generation},
  year = {2026},
  url = {https://github.com/IshCPU-VideoGenLab/codec-video-gen},
  institution = {University of Ghana, Legon}
}
```

---

## Contributing

See the [Contributing Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/CONTRIBUTING.md)
and [Version Control Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/VERSION_CONTROL_GUIDE.md).

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Phase 3 of [IshCPU-VideoGenLab](https://github.com/IshCPU-VideoGenLab). The most novel component — turning video generation into a codec problem.*
