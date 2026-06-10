#!/usr/bin/env python
"""Phase 3 probe: temporal redundancy of generated video (codec justification).

The codec design (generate keyframes, predict inter-frame deltas) is justified
only if consecutive generated frames are redundant. This generates a short
multi-frame video with the real Wan DiT, decodes it through the VAE, and measures
adjacent-frame similarity in latent and pixel space.

IMPORTANT: this uses a DUMMY text embedding (no T5), so the content is abstract
and temporally incoherent by construction -- it CANNOT validate the codec claim,
which requires real (prompted) generation. See docs/phase3_results.md. The probe
is kept to document that finding and to be re-run once a real text embedding is
available (precomputed once on a larger machine).

Usage:
    HF_TOKEN=... python scripts/temporal_redundancy.py
"""
import math

import numpy as np
import psutil
import torch
import torch.nn.functional as F

torch.set_num_threads(psutil.cpu_count(logical=False) or 4)
torch.manual_seed(0)
from diffusers import WanTransformer3DModel, AutoencoderKLWan  # noqa: E402

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
T = 3        # latent frames (T*256 tokens)
STEPS = 4


def main() -> None:
    dit = WanTransformer3DModel.from_pretrained(
        REPO, subfolder="transformer", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).eval()
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).eval()
    zmean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    zstd = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    txt = torch.randn(1, 512, 4096, dtype=torch.bfloat16)  # DUMMY — not a real prompt
    x = torch.randn(1, 16, T, 32, 32, dtype=torch.bfloat16)

    with torch.no_grad():
        for i in range(STEPS):
            t = torch.tensor([int((1 - i / STEPS) * 999)])
            x = x - (1 / STEPS) * dit(x, t, txt, return_dict=False)[0]
        lat = x.float()
        vid = vae.decode(lat * zstd + zmean, return_dict=False)[0]  # (1,3,Tp,256,256)

    print(f"latent frames T={T} -> decoded pixel frames Tp={vid.shape[2]}")
    print("\nLATENT adjacent-frame cosine (high = redundant):")
    for t in range(1, T):
        c = F.cosine_similarity(lat[:, :, t].flatten(), lat[:, :, t - 1].flatten(), dim=0).item()
        print(f"  frame {t-1}->{t}: cosine={c:.3f}")

    v = ((vid[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2).numpy()
    print("\nPIXEL adjacent-frame PSNR + delta-energy:")
    for t in range(1, vid.shape[2]):
        mse = float(np.mean((v[t] - v[t - 1]) ** 2))
        psnr = 99.0 if mse < 1e-9 else 20 * math.log10(1 / math.sqrt(mse))
        delta = float(np.linalg.norm(v[t] - v[t - 1]) / (np.linalg.norm(v[t]) + 1e-9))
        print(f"  frame {t-1}->{t}: PSNR={psnr:.1f} dB, delta-energy={delta*100:.1f}%")


if __name__ == "__main__":
    main()
