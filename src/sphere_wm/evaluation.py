"""Post-binarize CLIP-cosine evaluation across LatentWatermarker variants.

Given a trained encoder/decoder pair, train-set latents (for fitting
watermarker stats) and test-set CLIP embeddings, computes per-variant
CLIP cosine (mean ± std, plus per-class breakdown). Returns a nested
dict that is JSON-serializable.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from sphere_wm.config import CATEGORIES, DEVICE
from sphere_wm.models.watermarker import LatentWatermarker


@torch.no_grad()
def evaluate_variant(
    decode_fn: Callable[[np.ndarray], np.ndarray],
    train_latents: np.ndarray,
    test_latents: np.ndarray,
    test_clip: np.ndarray,
    test_labels: np.ndarray,
    variant: str,
) -> dict:
    wm = LatentWatermarker(
        latent_dim=train_latents.shape[1],
        watermark_bits=train_latents.shape[1],
        variant=variant,
    ).fit(train_latents)

    cos_per_sample = []
    for z, x in zip(test_latents, test_clip):
        bits = wm.latent_to_watermark(z)
        z_rec = wm.watermark_to_latent(bits)
        x_rec = decode_fn(z_rec[None, :])[0]
        x_rec = x_rec / (np.linalg.norm(x_rec) + 1e-12)
        x_n = x / (np.linalg.norm(x) + 1e-12)
        cos_per_sample.append(float(np.dot(x_n, x_rec)))
    cos = np.array(cos_per_sample)

    per_class = {}
    for i, cat in enumerate(CATEGORIES):
        m = test_labels == i
        if m.any():
            per_class[cat] = {
                "mean": float(cos[m].mean()),
                "std": float(cos[m].std()),
                "n": int(m.sum()),
            }

    return {
        "variant": variant,
        "n": int(len(cos)),
        "overall": {
            "mean": float(cos.mean()),
            "std": float(cos.std()),
            "min": float(cos.min()),
            "max": float(cos.max()),
        },
        "per_class": per_class,
    }


def make_decoder(model, device=DEVICE) -> Callable[[np.ndarray], np.ndarray]:
    @torch.no_grad()
    def _decode(z: np.ndarray) -> np.ndarray:
        zt = torch.from_numpy(z.astype(np.float32)).to(device)
        out = model.decode(zt)
        return out.cpu().numpy()

    return _decode
