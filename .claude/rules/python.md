# Python Rules — codec-video-gen

- Python 3.9: `List[str]` not `list[str]`, no `match` statements
- Type hints on ALL function signatures
- Google-style docstrings on all public functions
- `logging` module, never `print()` for production code
- Prefer functions + dataclasses over classes with methods
- `torch.no_grad()` for all inference
- float16 default, float32 only for numerically sensitive ops
- Delta predictor: keep under 200M params, target 50-100M
- GOP sizes: test with intervals of 4, 8, 16
- Files under 300 lines
- Tests: `pytest`, use small dimensions (d_model=64, frames=4-8)
- Absolute imports only
- No bare `except:` — catch specific exceptions
