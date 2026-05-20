"""Compute the honest comparison table.

For each trained model we:
  1. Extract train-set latents (μ) → fit watermarker stats
  2. Extract test-set latents (μ)
  3. For each binarization variant valid for that model, run the
     evaluate_variant pipeline and record overall + per-class cosine.

Outputs:
  results/comparison_table.json   — machine readable
  results/comparison_table.md     — human-readable Markdown for the README
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader

from sphere_wm.config import CHECKPOINT_DIR, DEVICE, RESULTS_DIR
from sphere_wm.data import CLIPFeatureDataset
from sphere_wm.evaluation import evaluate_variant, make_decoder
from sphere_wm.models.gaussian_vae import CLIPCompressionVAE
from sphere_wm.models.hyperspherical_vae import HypersphericalVAE
from sphere_wm.training import extract_latents
from sphere_wm.utils import load_checkpoint


def _eval_model(model_name, model, is_sphere, variants, results):
    train_loader = DataLoader(CLIPFeatureDataset("train"), batch_size=128, shuffle=False)
    test_loader = DataLoader(CLIPFeatureDataset("test"), batch_size=128, shuffle=False)
    train_z, _ = extract_latents(model, train_loader, device=DEVICE, is_sphere=is_sphere)
    test_z, test_y = extract_latents(model, test_loader, device=DEVICE, is_sphere=is_sphere)
    test_clip = CLIPFeatureDataset("test").features
    decoder = make_decoder(model, device=DEVICE)
    for v in variants:
        out = evaluate_variant(decoder, train_z, test_z, test_clip, test_y, variant=v)
        out["model"] = model_name
        print(f"[{model_name} / {v}] overall cos = {out['overall']['mean']:.4f}")
        results.append(out)


def main():
    results: list[dict] = []

    clip_vae = CLIPCompressionVAE(latent_dim=100).to(DEVICE)
    load_checkpoint(clip_vae, CHECKPOINT_DIR / "clip_vae.pt", map_location=DEVICE)
    _eval_model("clip_vae", clip_vae, is_sphere=False,
                variants=["clip_default"], results=results)

    sphere_vae = HypersphericalVAE(latent_dim=100).to(DEVICE)
    load_checkpoint(sphere_vae, CHECKPOINT_DIR / "sphere_vae.pt", map_location=DEVICE)
    _eval_model(
        "sphere_vae", sphere_vae, is_sphere=True,
        variants=["clip_default", "sphere_centered", "sphere_rescaled", "sphere_reprojected"],
        results=results,
    )

    (RESULTS_DIR / "comparison_table.json").write_text(json.dumps(results, indent=2))

    rows = [("Model", "Variant", "Overall cos", "Normal", "Violence", "Sexual")]
    for r in results:
        pc = r["per_class"]
        rows.append((
            r["model"], r["variant"], f"{r['overall']['mean']:.4f}",
            f"{pc.get('normal', {}).get('mean', float('nan')):.4f}",
            f"{pc.get('violence', {}).get('mean', float('nan')):.4f}",
            f"{pc.get('sexual', {}).get('mean', float('nan')):.4f}",
        ))
    header = "| " + " | ".join(rows[0]) + " |\n| " + " | ".join(["---"] * len(rows[0])) + " |\n"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    (RESULTS_DIR / "comparison_table.md").write_text(header + body + "\n")
    print("\nWrote results/comparison_table.{json,md}")


if __name__ == "__main__":
    main()
