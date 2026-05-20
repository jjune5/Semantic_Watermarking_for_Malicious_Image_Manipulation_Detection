import numpy as np
import pytest

from sphere_wm.models.watermarker import LatentWatermarker


def test_binarize_produces_bits():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((100, 100))
    w = LatentWatermarker(latent_dim=100, watermark_bits=100, variant="clip_default").fit(z)
    bits = w.latent_to_watermark(z[0])
    assert bits.shape == (100,)
    assert set(np.unique(bits).tolist()).issubset({0, 1})


def test_clip_default_round_trip_signs_match():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 100))
    w = LatentWatermarker(latent_dim=100, watermark_bits=100, variant="clip_default").fit(z)
    z0 = z[0]
    bits = w.latent_to_watermark(z0)
    rec = w.watermark_to_latent(bits)
    assert np.all((rec > 0) == (z0 > 0))


def test_sphere_centered_uses_train_mean():
    # Shift all latents far from origin: without centering every bit would be 1.
    rng = np.random.default_rng(0)
    z = rng.standard_normal((500, 100)) + 5.0
    w = LatentWatermarker(latent_dim=100, watermark_bits=100, variant="sphere_centered").fit(z)
    bits = w.latent_to_watermark(z[0])
    assert 0 < bits.sum() < 100


def test_sphere_rescaled_returns_train_scale():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((500, 100)) * 3.0 + 7.0
    w = LatentWatermarker(latent_dim=100, watermark_bits=100, variant="sphere_rescaled").fit(z)
    bits = w.latent_to_watermark(z[0])
    rec = w.watermark_to_latent(bits)
    assert rec.mean() > 3.0
    assert rec.std() > 1.0


def test_sphere_reprojected_outputs_unit_norm():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((500, 100))
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    w = LatentWatermarker(latent_dim=100, watermark_bits=100, variant="sphere_reprojected").fit(z)
    rec = w.watermark_to_latent(w.latent_to_watermark(z[0]))
    assert abs(np.linalg.norm(rec) - 1.0) < 1e-5


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        LatentWatermarker(latent_dim=100, watermark_bits=100, variant="???")
