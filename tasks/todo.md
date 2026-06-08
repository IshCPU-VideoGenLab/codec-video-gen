# Phase 3 — codec-video-gen Task Roadmap

> Claude Code: check this file at the start of every session.

---

## Milestone 1: Temporal Redundancy Analysis
- [ ] Load Wan 1.3B and generate latent frames for sample videos
- [ ] Compute frame-to-frame cosine similarity in latent space
- [ ] Compute frame-to-frame MSE in latent space
- [ ] Compute temporal autocorrelation across frame sequences
- [ ] Quantify: what % of information is shared between consecutive frames?
- [ ] Produce charts showing redundancy vs frame distance
- [ ] Document findings — this motivates the entire codec approach

## Milestone 2: GOP Scheduler
- [ ] Implement `GOPScheduler` — assigns I/P frame types given total frames + interval
- [ ] Support configurable keyframe intervals (4, 8, 16, 32)
- [ ] Support forced I-frames at scene boundaries (future extension)
- [ ] Implement compute cost estimator: given GOP structure, estimate total FLOPs
- [ ] Unit test: correct frame type assignment for various configurations
- [ ] Unit test: cost estimation matches expected savings

## Milestone 3: Delta Predictor Network
- [ ] Design delta predictor architecture (small ConvNet + residual blocks)
- [ ] Implement `DeltaPredictor` module (target: 50-100M params)
- [ ] Input: previous frame latent + timestep embedding
- [ ] Output: predicted delta (change from previous frame)
- [ ] Verify: output shape matches latent frame shape
- [ ] Verify: parameter count within budget
- [ ] Unit test: forward pass, gradient flow, float16 stability
- [ ] Benchmark: single delta prediction time on CPU

## Milestone 4: Keyframe Generation Wrapper
- [ ] Implement `KeyframeGenerator` — wraps the full model for I-frame generation
- [ ] Support using original Wan 1.3B or Phase 2 modified (Mamba) model
- [ ] Memory-efficient: load/unload full model around keyframe generation
- [ ] Test: generates valid latent frames

## Milestone 5: Codec Pipeline
- [ ] Implement `CodecPipeline` — end-to-end generation using GOP structure
- [ ] For each frame: dispatch to keyframe generator (I) or delta predictor (P)
- [ ] P-frame generation: prev_frame + delta_predictor(prev_frame, t)
- [ ] Handle error accumulation: optional correction at GOP boundaries
- [ ] Implement quality-aware mode: fall back to I-frame if delta is too large
- [ ] Integration test: full pipeline generates N frames

## Milestone 6: Evaluation & Comparison
- [ ] Generate same video with all-I-frame baseline vs codec pipeline
- [ ] Measure wall-clock time: baseline vs codec at various GOP sizes
- [ ] Measure quality: MSE, PSNR, cosine similarity frame-by-frame
- [ ] Produce the key chart: quality vs GOP size (keyframe interval)
- [ ] Produce the key chart: speedup vs GOP size
- [ ] Compute theoretical FLOPs savings
- [ ] Analyze error drift: does quality degrade across P-frames within a GOP?

## Milestone 7: Documentation & Polish
- [ ] Write `docs/codec_design.md` with full methodology
- [ ] Update README with actual results
- [ ] Clean code, docstrings, type hints
- [ ] All tests pass
- [ ] Tag v0.1.0
- [ ] Write Phase 4 (bitnet-video) handoff summary

---

## Notes
- Milestone 1 is critical — if redundancy is low, the codec approach has less value.
  (But video IS highly redundant, so this should confirm the hypothesis.)
- Milestone 3 (delta predictor) is the most research-sensitive piece.
  Start with a simple ConvNet and iterate.
- The key paper figure comes from Milestone 6: the quality-speedup tradeoff.
