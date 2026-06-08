# Codec-Inspired Video Generation Design

## The Central Insight

Video compression codecs (H.264, H.265, AV1) run in real time on CPUs by
exploiting a simple fact: consecutive video frames are nearly identical.
A person walking, a camera panning — frame-to-frame, only a small fraction
of pixels change.

Current video diffusion models ignore this completely. Every frame passes
through the same expensive denoising pipeline regardless of similarity to
its neighbors. This is equivalent to JPEG-compressing every frame of a
movie independently — technically correct but enormously wasteful.

**Our approach: treat video generation as a codec problem.**

## How Video Codecs Work

### Frame Types

- **I-frame (Intra)**: Compressed independently. Contains full image data.
  The most expensive frame type to encode/decode.

- **P-frame (Predicted)**: Compressed as a delta from the previous
  reconstructed frame. Only stores what changed. Typically 10-50× smaller
  than an I-frame.

- **B-frame (Bidirectional)**: Uses both past and future frames as
  reference. Even more efficient but adds latency. We omit B-frames for
  simplicity in this initial design.

### GOP (Group of Pictures)

A GOP is the sequence from one I-frame to the next:

```
I P P P P P P P I P P P P P P P I ...
└── GOP 1 ──────┘└── GOP 2 ──────┘
```

Typical GOP size: 8-30 frames. Larger GOPs = more compression but more
risk of error accumulation.

## Applying Codec Design to Generation

### Standard Video Diffusion (Baseline)

```
Frame 0: Full model → latent → decode → frame
Frame 1: Full model → latent → decode → frame
Frame 2: Full model → latent → decode → frame
...
Frame N: Full model → latent → decode → frame

Cost: N × full_model_cost
```

### Codec-Style Video Generation (Ours)

```
Frame 0 [I]: Full model → latent₀
Frame 1 [P]: Delta predictor(latent₀) → delta₁ → latent₀ + delta₁ = latent₁
Frame 2 [P]: Delta predictor(latent₁) → delta₂ → latent₁ + delta₂ = latent₂
...
Frame 7 [P]: Delta predictor(latent₆) → delta₇ → latent₆ + delta₇ = latent₇
Frame 8 [I]: Full model → latent₈   (new keyframe, resets error)
Frame 9 [P]: Delta predictor(latent₈) → delta₉ → ...

Cost: K × full_model_cost + (N-K) × delta_cost
      where K = num_keyframes, delta_cost ≈ 0.05-0.1 × full_model_cost
```

### Theoretical Savings

For 16 frames with keyframe interval 8:
- Baseline: 16 × 1.0 = 16.0 units
- Codec: 2 × 1.0 + 14 × 0.1 = 3.4 units
- **Speedup: 4.7×**

For 32 frames with keyframe interval 16:
- Baseline: 32 × 1.0 = 32.0 units
- Codec: 2 × 1.0 + 30 × 0.1 = 5.0 units
- **Speedup: 6.4×**

## The Delta Predictor

### Design Constraints

1. **Must be much smaller than the full model** — 50-100M params vs 1.3B
2. **Must run fast on CPU** — the whole point is speed
3. **Must predict plausible frame-to-frame changes** — not arbitrary noise
4. **Must work in latent space** — we predict deltas in the VAE latent
   space, not pixel space

### Architecture

```
prev_latent (4, H, W)
    │
    Conv2d(4 → 128, 3×3)     # Input projection
    │
    ResBlock(128, time_cond)   # Residual block with timestep conditioning
    ResBlock(128, time_cond)
    ResBlock(128, time_cond)
    ResBlock(128, time_cond)
    │
    GroupNorm → SiLU
    │
    Conv2d(128 → 4, 3×3)     # Output projection (initialized near-zero)
    │
    delta (4, H, W)
```

The output layer is initialized to produce near-zero output. This means:
- At initialization, `next_frame ≈ prev_frame + 0 = prev_frame`
- The model starts as an identity function and learns to deviate
- This is crucial for stability — random deltas would destroy quality

### Timestep Conditioning

The delta predictor receives a timestep signal indicating the frame's
position. This allows it to modulate its predictions:
- Early P-frames (close to keyframe): small deltas
- Late P-frames (far from keyframe): potentially larger deltas

## Error Accumulation

The main risk of codec-style generation: each P-frame's error compounds.
If frame t has error ε, frame t+1 predicts from a corrupted reference,
adding its own error. Over a full GOP, errors can drift significantly.

### Mitigation Strategies

1. **Periodic I-frames** (built-in): Every GOP starts fresh, resetting
   accumulated error. Shorter GOPs = less drift, more compute.

2. **Error correction at boundaries**: Before generating the next GOP,
   apply a correction pass to the last P-frame to reduce drift.

3. **Quality-aware mode**: Monitor delta norm during generation. If a
   delta is unusually large, fall back to I-frame generation for that
   frame. This adaptively places keyframes where the scene changes most.

4. **Gradient-based correction** (future work): Fine-tune P-frames
   against the full model's distribution to reduce systematic bias.

## Key Measurements

### What We Measure

1. **Temporal redundancy**: How similar are consecutive latent frames?
   If redundancy is 95%, the delta predictor only needs to capture 5%.

2. **Delta predictor accuracy**: How well does the predicted delta match
   the true frame-to-frame change?

3. **Error drift**: How does quality degrade across P-frames within a GOP?

4. **Optimal GOP size**: The sweet spot between compute savings and quality.

### Expected Results

Based on the nature of video:
- **Temporal redundancy**: 90-98% for typical content
- **Optimal GOP size**: 8-16 frames
- **Speedup**: 3-6× for typical configurations
- **Quality loss**: <5% degradation for reasonable GOP sizes

## Integration with Other Phases

- **Phase 2** (Mamba): The full model used for I-frames can be the
  Mamba-modified version, stacking speedups multiplicatively.
- **Phase 4** (BitNet): Both the full model AND the delta predictor
  can be quantized to 1-bit, further reducing compute.
- **Phase 5** (AVX2): The delta predictor's Conv2D operations are
  straightforward to implement in AVX2 kernels.

Combined effect: each optimization multiplies with the others.
