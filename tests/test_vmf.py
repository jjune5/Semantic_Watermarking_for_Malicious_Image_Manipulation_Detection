import torch

from sphere_wm.vmf import (
    HypersphericalUniform,
    VonMisesFisher,
    ive,
    kl_vmf_uniform,
)


def test_uniform_sample_is_unit_norm():
    d = 99
    u = HypersphericalUniform(dim=d, device="cpu")
    s = u.sample(torch.Size([1024]))
    assert s.shape == (1024, d + 1)
    norms = s.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_vmf_sample_is_unit_norm():
    torch.manual_seed(0)
    loc = torch.tensor([[1.0] + [0.0] * 99])
    scale = torch.tensor([[10.0]])
    q = VonMisesFisher(loc=loc, scale=scale)
    z = q.rsample(torch.Size([256]))
    z_flat = z.view(-1, 100)
    norms = z_flat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_vmf_concentrates_near_loc_for_high_kappa():
    # vMF on S^(d-1) with concentration κ has E[cos(z, loc)] ≈ 1 - (d-1)/(2κ).
    # For d=100, we need κ ≥ ~2000 to drive E[cos] above 0.95.
    torch.manual_seed(0)
    loc = torch.tensor([[1.0] + [0.0] * 99])
    scale = torch.tensor([[2000.0]])
    q = VonMisesFisher(loc=loc, scale=scale)
    z = q.rsample(torch.Size([512])).view(-1, 100)
    cos = (z * loc).sum(dim=-1)
    assert cos.mean() > 0.95


def test_ive_positive_for_positive_z():
    z = torch.tensor([0.1, 1.0, 10.0])
    out = ive(0, z)
    assert (out > 0).all()


def test_kl_is_finite_and_nonneg_for_high_kappa():
    loc = torch.tensor([[1.0] + [0.0] * 99])
    scale = torch.tensor([[50.0]])
    q = VonMisesFisher(loc=loc, scale=scale)
    p = HypersphericalUniform(dim=99, device="cpu")
    kl = kl_vmf_uniform(q, p)
    assert torch.isfinite(kl).all()
    assert kl.item() > -1e-3
