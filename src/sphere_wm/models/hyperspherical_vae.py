"""Hyperspherical VAE — port of cell 16 in colab_codes/ablation_sphere_vae.ipynb.

Encoder maps a CLIP embedding (R^input_dim) to a vMF distribution on
S^(latent_dim - 1): a unit-norm mean direction μ and a positive concentration κ.
Decoder maps a sampled latent back to R^input_dim and L2-normalizes the
output so reconstruction and target both live on the unit sphere.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sphere_wm.vmf import HypersphericalUniform, VonMisesFisher


class HypersphericalVAE(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, latent_dim: int = 100):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.fc_e0 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc_e1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_kappa = nn.Linear(hidden_dim, 1)

        self.fc_d0 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim * 2, input_dim)

    def encode(self, x):
        h = F.relu(self.fc_e0(x))
        h = F.relu(self.fc_e1(h))
        z_mean = self.fc_mean(h)
        z_mean = z_mean / (z_mean.norm(dim=-1, keepdim=True) + 1e-8)
        z_kappa = F.softplus(self.fc_kappa(h)) + 1.0
        return z_mean, z_kappa

    def decode(self, z):
        h = F.relu(self.fc_d0(z))
        h = F.relu(self.fc_d1(h))
        recon = self.fc_out(h)
        return F.normalize(recon, p=2, dim=-1)

    def reparameterize(self, z_mean, z_kappa):
        q_z = VonMisesFisher(z_mean, z_kappa)
        p_z = HypersphericalUniform(self.latent_dim - 1, device=z_mean.device)
        return q_z, p_z

    def forward(self, x):
        z_mean, z_kappa = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_kappa)
        z = q_z.rsample()
        recon = self.decode(z)
        return recon, z, z_mean, z_kappa, q_z, p_z
