"""Shared train / validate loops for both models.

Trains on PRE-COMPUTED CLIP embeddings (no online CLIP forward).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from sphere_wm.config import BETA, DEVICE


def train_one_epoch_gaussian(model, loader, optimizer, loss_fn,
                             beta=BETA, device=DEVICE):
    model.train()
    stats = defaultdict(list)
    for emb, _ in tqdm(loader, desc="train", leave=False):
        emb = emb.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, _ = model(emb)
        loss, recon_l, kld_l = loss_fn(recon, emb, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        stats["total"].append(loss.item())
        stats["recon"].append(recon_l.item())
        stats["kld"].append(kld_l.item())
    return {k: float(np.mean(v)) for k, v in stats.items()}


@torch.no_grad()
def validate_gaussian(model, loader, loss_fn, beta=BETA, device=DEVICE):
    model.eval()
    stats = defaultdict(list)
    all_cos = []
    for emb, _ in loader:
        emb = emb.to(device)
        recon, mu, logvar, _ = model(emb)
        loss, recon_l, kld_l = loss_fn(recon, emb, mu, logvar, beta)
        stats["total"].append(loss.item())
        stats["recon"].append(recon_l.item())
        stats["kld"].append(kld_l.item())
        cos = torch.nn.functional.cosine_similarity(recon, emb, dim=1)
        all_cos.extend(cos.cpu().numpy().tolist())
    return ({k: float(np.mean(v)) for k, v in stats.items()},
            float(np.mean(all_cos)))


def train_one_epoch_sphere(model, loader, optimizer, loss_fn,
                           beta=BETA, device=DEVICE):
    model.train()
    stats = defaultdict(list)
    for emb, _ in tqdm(loader, desc="train", leave=False):
        emb = emb.to(device)
        optimizer.zero_grad()
        recon, z, z_mean, z_kappa, q_z, p_z = model(emb)
        loss, recon_l, kl_l = loss_fn(recon, emb, q_z, p_z, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        stats["total"].append(loss.item())
        stats["recon"].append(recon_l.item())
        stats["kl"].append(kl_l.item())
    return {k: float(np.mean(v)) for k, v in stats.items()}


@torch.no_grad()
def validate_sphere(model, loader, loss_fn, beta=BETA, device=DEVICE):
    model.eval()
    stats = defaultdict(list)
    all_cos = []
    for emb, _ in loader:
        emb = emb.to(device)
        recon, z, z_mean, z_kappa, q_z, p_z = model(emb)
        loss, recon_l, kl_l = loss_fn(recon, emb, q_z, p_z, beta)
        stats["total"].append(loss.item())
        stats["recon"].append(recon_l.item())
        stats["kl"].append(kl_l.item())
        cos = torch.nn.functional.cosine_similarity(recon, emb, dim=-1)
        all_cos.extend(cos.cpu().numpy().tolist())
    return ({k: float(np.mean(v)) for k, v in stats.items()},
            float(np.mean(all_cos)))


@torch.no_grad()
def extract_latents(model, loader, device=DEVICE, is_sphere: bool = False):
    """Return (latents [N, D], labels [N]) using μ (deterministic encoding)."""
    model.eval()
    Z, Y = [], []
    for emb, lbl in loader:
        emb = emb.to(device)
        if is_sphere:
            z_mean, _ = model.encode(emb)
            Z.append(z_mean.cpu().numpy())
        else:
            mu, _ = model.encode(emb)
            Z.append(mu.cpu().numpy())
        Y.append(lbl.numpy() if isinstance(lbl, torch.Tensor) else np.asarray(lbl))
    return np.concatenate(Z, axis=0), np.concatenate(Y, axis=0)
