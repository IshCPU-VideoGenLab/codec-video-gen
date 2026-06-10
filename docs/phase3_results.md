# Phase 3 Results — Temporal Redundancy (and why it needs real generation)

The codec design generates keyframes (I-frames) at full fidelity and predicts
inter-frame deltas (P-frames) cheaply. This is justified **only if consecutive
generated frames are redundant**. We probed that on the real Wan DiT + VAE
(`scripts/temporal_redundancy.py`).

## What we measured (dummy text)

Short multi-frame generation (3 latent frames → 9 decoded frames), adjacent-frame
similarity:

| space | metric | values |
|-------|--------|--------|
| latent | cosine (adjacent) | 0.28, 0.29 |
| pixel | PSNR (adjacent) | 11–16 dB |
| pixel | delta-energy | 32–52% |

The frames are **temporally incoherent** — adjacent frames differ by ~40%.

## The key finding: this can't validate the codec design

This generation used a **dummy text embedding** (the ~11 GB T5 is not loaded).
With no real conditioning, the model has no coherent scene to render, so the
output is temporally noisy **by construction**. Its frame-to-frame redundancy
says nothing about *real* generated video.

**Phase 3 is the first pillar that cannot be validated with dummy embeddings.**
Phases 1, 2, and 4 concern the model's *computation* (where compute goes; how
faithful the output stays under surgery/quantization) — dummy text is fine there.
Phase 3 concerns the *content's temporal structure*, which requires **real,
prompted generation**.

## What's true regardless

- The codec's **compute saving is structural**: generating K-1 of every K frames
  as deltas through a small predictor means K-1 fewer full DiT forward passes,
  independent of content. Phase 1 already showed a single full forward is ~5 s on
  CPU, so fewer full forwards is a direct, real win.
- The **quality** of that scheme depends on temporal redundancy, which is
  well-established for real video (this is exactly what H.264/H.265 exploit) but
  must be confirmed for *this model's* real output.

## Next

Validate on real generation:
1. Precompute a real text embedding once (load T5 on a larger machine / free
   Colab; cache the ~4 MB tensor) — avoids ever loading the 22 GB T5 on the CPU
   target.
2. Re-run this probe with the real embedding; expect high adjacent-frame
   redundancy for a coherent prompted scene.
3. Then train/evaluate the P-frame delta predictor against real keyframes.
