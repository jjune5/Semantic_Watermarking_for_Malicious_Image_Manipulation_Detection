"""β-VAE on CLIP embeddings (Gaussian latent) — port of cell 25 of
colab_codes/clip_vae.ipynb (and the clip_vae_with_binarization reference).

Encoder maps R^input_dim -> R^latent_dim with diagonal Gaussian (μ, logσ²).
Decoder reconstructs to R^input_dim and L2-normalizes so cosine distance is
the natural reconstruction objective."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPCompressionVAE(nn.Module):
    def __init__(self, input_dim: int = 512, latent_dim: int = 100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return F.normalize(self.decoder(z), p=2, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
