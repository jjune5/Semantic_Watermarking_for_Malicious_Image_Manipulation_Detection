"""Train HypersphericalVAE (vMF) on pre-computed CLIP features.

Same architecture/data/loss/β/epochs/batch as scripts/train_clip_vae.py.
The only deliberate hyperparameter divergence is the learning rate:
clip_vae uses 1e-3 (paper Table 1 / clip_vae.ipynb cell 30) and sphere_vae
uses 1e-4 (ablation_sphere_vae.ipynb cell 24). vMF gradients flow through
rejection sampling and a softplus(κ) head, both of which destabilize at
lr=1e-3 (κ collapses to ≈1, vMF degenerates toward Uniform). Both rates
match what each model's source notebook used; no extra tuning was done.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch.optim as optim

from sphere_wm.config import (BETA, CHECKPOINT_DIR, DEVICE,
                              NUM_EPOCHS, RESULTS_DIR, SEED)
from sphere_wm.data import make_loaders
from sphere_wm.losses import hyperspherical_vae_loss
from sphere_wm.models.hyperspherical_vae import HypersphericalVAE
from sphere_wm.training import train_one_epoch_sphere, validate_sphere
from sphere_wm.utils import save_checkpoint, set_seed

LEARNING_RATE = float(os.environ.get("LR", 1e-4))


def main():
    set_seed(SEED)
    print(f"[sphere_vae] device={DEVICE} epochs={NUM_EPOCHS} lr={LEARNING_RATE} beta={BETA}")
    train_loader, test_loader = make_loaders()
    model = HypersphericalVAE(latent_dim=100).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    history = {"train": [], "val": [], "val_cos": [], "lr": []}
    best_cos = -1.0
    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        tr = train_one_epoch_sphere(model, train_loader, opt, hyperspherical_vae_loss, BETA, DEVICE)
        va, cos = validate_sphere(model, test_loader, hyperspherical_vae_loss, BETA, DEVICE)
        sched.step(va["total"])
        history["train"].append(tr)
        history["val"].append(va)
        history["val_cos"].append(cos)
        history["lr"].append(opt.param_groups[0]["lr"])
        print(
            f"epoch {epoch:3d}/{NUM_EPOCHS}  "
            f"train recon {tr['recon']:.4f}  val recon {va['recon']:.4f}  val cos {cos:.4f}"
        )
        if cos > best_cos:
            best_cos = cos
            save_checkpoint(
                model,
                CHECKPOINT_DIR / "sphere_vae.pt",
                extra={"epoch": epoch, "val_cos": cos},
            )
    print(f"[sphere_vae] done in {time.time()-t0:.1f}s  best val cos = {best_cos:.4f}")
    (RESULTS_DIR / "sphere_vae_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
