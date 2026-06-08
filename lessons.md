# Lessons Learned

> Claude Code checks this before writing new code.

---

<!-- Example:
## 2026-07-01 — Delta predictor too large
**What happened:** Used 500M param delta predictor. OOM when loaded alongside full model.
**Rule:** Delta predictor must be ≤ 200M params. Target 50-100M for Pentium Gold.
-->
