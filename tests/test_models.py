import torch

from sphere_wm.models.hyperspherical_vae import HypersphericalVAE
from sphere_wm.models.gaussian_vae import CLIPCompressionVAE


def test_hyperspherical_vae_forward_shapes():
    model = HypersphericalVAE(input_dim=512, hidden_dim=256, latent_dim=100)
    model.eval()
    x = torch.randn(8, 512)
    x = x / x.norm(dim=-1, keepdim=True)
    recon, z, z_mean, z_kappa, q_z, p_z = model(x)
    assert recon.shape == (8, 512)
    assert z.shape == (8, 100)
    assert z_mean.shape == (8, 100)
    assert z_kappa.shape == (8, 1)


def test_hyperspherical_vae_decoder_unit_norm():
    model = HypersphericalVAE(latent_dim=100)
    model.eval()
    z = torch.randn(16, 100)
    z = z / z.norm(dim=-1, keepdim=True)
    out = model.decode(z)
    assert torch.allclose(out.norm(dim=-1), torch.ones(16), atol=1e-5)


def test_hyperspherical_vae_encoder_unit_mean():
    model = HypersphericalVAE(latent_dim=100)
    model.eval()
    x = torch.randn(4, 512)
    z_mean, z_kappa = model.encode(x)
    assert torch.allclose(z_mean.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert (z_kappa > 0).all()


def test_gaussian_vae_forward_shapes():
    m = CLIPCompressionVAE(input_dim=512, latent_dim=100)
    m.eval()
    x = torch.randn(8, 512)
    x = x / x.norm(dim=-1, keepdim=True)
    recon, mu, logvar, z = m(x)
    assert recon.shape == (8, 512)
    assert mu.shape == (8, 100)
    assert logvar.shape == (8, 100)
    assert z.shape == (8, 100)


def test_gaussian_vae_decoder_unit_norm():
    m = CLIPCompressionVAE(latent_dim=100)
    m.eval()
    z = torch.randn(16, 100)
    out = m.decode(z)
    assert torch.allclose(out.norm(dim=-1), torch.ones(16), atol=1e-5)
