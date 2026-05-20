"""Loss functions for both VAEs.

Both use 1 − cos(recon, target) + β · KL form, but with different KL terms:
  - Gaussian: closed-form KL(N(μ, σ²) ‖ N(0, I))
  - vMF:     -H(q) + H(p) on hypersphere
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sphere_wm.vmf import kl_vmf_uniform


def gaussian_vae_loss(recon, target, mu, logvar, beta: float = 0.01):
    cos = F.cosine_similarity(recon, target, dim=1).mean()
    recon_loss = 1 - cos
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss, recon_loss, kld_loss


def hyperspherical_vae_loss(recon, target, q_z, p_z, beta: float = 0.01):
    cos = F.cosine_similarity(recon, target, dim=-1)
    recon_loss = (1 - cos).mean()
    kl_loss = kl_vmf_uniform(q_z, p_z).mean()
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
