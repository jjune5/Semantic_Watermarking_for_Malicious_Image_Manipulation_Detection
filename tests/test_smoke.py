"""End-to-end smoke: synthesize 128 unit-norm 'CLIP' vectors, train both
models for 1 epoch, evaluate all variants. Asserts shapes/ranges only —
actual numbers are not meaningful at this scale."""
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sphere_wm.evaluation import evaluate_variant, make_decoder
from sphere_wm.losses import gaussian_vae_loss, hyperspherical_vae_loss
from sphere_wm.models.gaussian_vae import CLIPCompressionVAE
from sphere_wm.models.hyperspherical_vae import HypersphericalVAE
from sphere_wm.training import (
    extract_latents,
    train_one_epoch_gaussian,
    train_one_epoch_sphere,
    validate_gaussian,
    validate_sphere,
)


def _toy_loader(n=128, batch=32):
    torch.manual_seed(0)
    x = torch.randn(n, 512)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True, drop_last=True)


def test_smoke_gaussian():
    m = CLIPCompressionVAE(latent_dim=100)
    opt = optim.Adam(m.parameters(), lr=1e-3)
    loader = _toy_loader()
    train_one_epoch_gaussian(m, loader, opt, gaussian_vae_loss, device="cpu")
    _, cos = validate_gaussian(m, loader, gaussian_vae_loss, device="cpu")
    assert -1.0 <= cos <= 1.0


def test_smoke_sphere():
    m = HypersphericalVAE(latent_dim=100)
    opt = optim.Adam(m.parameters(), lr=1e-3)
    loader = _toy_loader()
    train_one_epoch_sphere(m, loader, opt, hyperspherical_vae_loss, device="cpu")
    _, cos = validate_sphere(m, loader, hyperspherical_vae_loss, device="cpu")
    assert -1.0 <= cos <= 1.0


def test_smoke_evaluate_pipeline():
    m = HypersphericalVAE(latent_dim=100)
    m.eval()
    torch.manual_seed(0)
    n = 128
    x = torch.randn(n, 512)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.zeros(n, dtype=torch.long)
    # shuffle=False so the loader order matches the underlying tensor order
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=False)
    train_z, _ = extract_latents(m, loader, device="cpu", is_sphere=True)
    test_z, test_y = extract_latents(m, loader, device="cpu", is_sphere=True)
    res = evaluate_variant(
        make_decoder(m, device="cpu"),
        train_z, test_z, x.numpy(), test_y,
        variant="sphere_rescaled",
    )
    assert "overall" in res and "per_class" in res
    assert -1.0 <= res["overall"]["mean"] <= 1.0
