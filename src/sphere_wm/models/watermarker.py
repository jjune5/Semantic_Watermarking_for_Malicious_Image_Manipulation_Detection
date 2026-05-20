"""Sign-based binarization with multiple recovery strategies.

Variants (set via __init__ ``variant=`` argument):

* ``clip_default``       — bit = (z > 0); recover = 2*bit - 1
* ``sphere_centered``    — bit = (z - μ > 0); recover = 2*bit - 1
* ``sphere_rescaled``    — bit = (z - μ > 0); recover = (2*bit - 1) * σ + μ
* ``sphere_reprojected`` — same as sphere_rescaled, then L2-normalize to sphere

In every variant, if ``latent_dim > watermark_bits`` a PCA is fit so the bit
budget exactly matches ``watermark_bits``. (Defaults: 100 == 100, no PCA.)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.decomposition import PCA

Variant = Literal[
    "clip_default", "sphere_centered", "sphere_rescaled", "sphere_reprojected"
]
_VALID = {"clip_default", "sphere_centered", "sphere_rescaled", "sphere_reprojected"}


class LatentWatermarker:
    def __init__(
        self,
        latent_dim: int = 100,
        watermark_bits: int = 100,
        variant: Variant = "clip_default",
    ):
        if variant not in _VALID:
            raise ValueError(
                f"unknown variant {variant!r}; expected one of {sorted(_VALID)}"
            )
        self.latent_dim = latent_dim
        self.watermark_bits = watermark_bits
        self.variant = variant
        self.pca: PCA | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, latent_vectors: np.ndarray) -> "LatentWatermarker":
        if self.latent_dim > self.watermark_bits:
            self.pca = PCA(n_components=self.watermark_bits)
            self.pca.fit(latent_vectors)
            projected = self.pca.transform(latent_vectors)
        else:
            projected = latent_vectors
        self.mean_ = projected.mean(axis=0)
        self.std_ = projected.std(axis=0)
        return self

    def _project(self, z: np.ndarray) -> np.ndarray:
        if self.pca is None:
            return z
        if z.ndim == 1:
            return self.pca.transform(z.reshape(1, -1))[0]
        return self.pca.transform(z)

    def _inverse_project(self, z: np.ndarray) -> np.ndarray:
        if self.pca is None:
            return z
        if z.ndim == 1:
            return self.pca.inverse_transform(z.reshape(1, -1))[0]
        return self.pca.inverse_transform(z)

    def latent_to_watermark(self, latent: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "call .fit() first"
        z = self._project(latent)
        if self.variant == "clip_default":
            return (z > 0).astype(int)
        return (z - self.mean_ > 0).astype(int)

    def watermark_to_latent(self, bits: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "call .fit() first"
        signs = bits.astype(float) * 2.0 - 1.0
        if self.variant in ("sphere_rescaled", "sphere_reprojected"):
            rec_projected = signs * self.std_ + self.mean_
        else:
            rec_projected = signs
        rec = self._inverse_project(rec_projected)
        if self.variant == "sphere_reprojected":
            rec = rec / (np.linalg.norm(rec, axis=-1, keepdims=True) + 1e-12)
        return rec
