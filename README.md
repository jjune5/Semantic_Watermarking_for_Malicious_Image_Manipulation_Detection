# CLIP-VAE & SDA-Net: Distribution-Based Semantic Watermarking

The framework embeds a recoverable semantic reference into an image as a 100-bit watermark
(**CLIP-VAE**) and exposes the direction of any post-hoc manipulation
through a lightweight prototype module (**SDA-Net**).

This repository extends the original submission with one additional
ablation that **swaps the Gaussian VAE for a hyperspherical (vMF) VAE**
so that the 100-D latent lives directly on the unit sphere `S^99`.

---

## Pipeline

![Pipeline overview](figures/pipeline_overview.png)

Raw images → CLIP-ViT-B/32 (512-D) → VAE (100-D latent) → sign-based
binarization → 100-bit watermark (channel-aware training with random
bit-flip noise) → SDA-Net prototypes → per-class distance scores +
direction-of-drift signal.

---

## Repository layout

```
colab_codes/
  clip_vae.ipynb              # Main: Gaussian β-VAE on CLIP embeddings
  SDA_net.ipynb               # Prototype-based drift detection on the latent
  baseline_exp.ipynb          # 5-way bit-flip comparison (SimHash, ITQ, HashNet, …)
  ablation_beta_vae.ipynb     # KL weight β sweep
  ablation_vqvae.ipynb        # VQ-VAE vs VAE
  ablation_dimensions.ipynb   # Latent dimensionality sweep
  ablation_inference.ipynb    # Inference-time settings
  ablation_sphere_vae.ipynb   # Hyperspherical (vMF) latent on S^99 ablation
preprocessing/
  prepare_dataset.py                          # Build dataset/{train,test}/{normal,sexual,violence}/
  extract_clip_embeddings.py                  # CLIP-ViT-B/32 → embeddings/*.npy (aspect-preserving)
  extract_clip_embeddings_notebook_style.py   # Bit-for-bit reproduction of the notebooks' CLIP path
src/sphere_wm/
  vmf.py                      # IveFunction, HypersphericalUniform, VonMisesFisher, kl_vmf_uniform
  models/
    hyperspherical_vae.py     # HypersphericalVAE (with kappa_init knob)
    gaussian_vae.py           # CLIPCompressionVAE (Gaussian baseline)
    watermarker.py            # LatentWatermarker — 4 binarization variants
  data.py, losses.py, training.py, evaluation.py, config.py, utils.py
scripts/
  train_clip_vae.py           # Gaussian VAE — 30 epoch CPU/GPU trainer
  train_sphere_vae.py         # vMF VAE — same trainer with kappa_init=50 + β=0 defaults
  evaluate_binarization.py    # 4 binarize variants → results/comparison_table.{md,json}
tests/                        # pytest suite (19 tests, all green)
embeddings/                   # Pre-computed CLIP features (~16 MB)
figures/                      # Plots from notebook runs + pipeline_overview.png
requirements.txt
```

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/Shilin-LU/VINE.git   # external watermarking backbone
```

**Dataset.** Three public sources (`normal`: Kaggle News images, `violence`:
HOD benchmark, `sexual`: Figshare adult-content set). Image data is
**not** redistributed — see the paper's Reproducibility Statement.
After downloading the three sources into `raw_images/{normal,violence,sexual}/`:

```bash
python preprocessing/prepare_dataset.py --input-dir ./raw_images --output-dir ./dataset --seed 42
python preprocessing/extract_clip_embeddings.py --dataset-dir ./dataset --output-dir ./embeddings
```

| Class | Total | Train | Test |
|---|---:|---:|---:|
| `normal` | 4,000 | 3,200 | 800 |
| `violence` | 2,002 | 1,601 | 401 |
| `sexual` | 2,000 | 1,600 | 400 |

Pre-computed 512-D CLIP embeddings are shipped under `embeddings/` so
the SDA-Net and most ablations can be re-run without redownloading the
images.

---

## Notebook reference

| Notebook | Stage | Needs raw images? | Runtime on T4 |
|---|---|:---:|---:|
| `clip_vae.ipynb` | 1. Train β-VAE | ✅ | ~25 min |
| `SDA_net.ipynb` | 2. Train SDA-Net | — (uses embeddings) | ~15 min |
| `baseline_exp.ipynb` | 3. Bit-flip comparison | partial | ~90 min |
| `ablation_beta_vae.ipynb` | Ablation | — | ~45 min |
| `ablation_vqvae.ipynb` | Ablation | — | ~30 min |
| `ablation_dimensions.ipynb` | Ablation | — | ~50 min |
| `ablation_inference.ipynb` | Ablation | — | ~20 min |
| `ablation_sphere_vae.ipynb` | Ablation (new) | ✅ | ~30 min |

All notebooks ship with cleared outputs; rendered figures from a known-good
run are in `figures/`.

---

## Hyperspherical-latent extension (`ablation_sphere_vae.ipynb` + `scripts/train_sphere_vae.py`)

**What changes vs `clip_vae.ipynb`.** The Gaussian VAE
(`z = μ + σ ⊙ ε`, prior `N(0, I)`, KL closed-form) is replaced by a
**von Mises–Fisher VAE** (`z ∼ vMF(μ, κ)` on `S^99`, prior uniform on
the hypersphere, KL computed against `Uniform(S^99)`). The encoder
output is a unit-norm mean direction `μ` and a positive concentration
`κ`; the decoder L2-normalizes the reconstructed CLIP embedding to
keep both ends of the loop on the unit sphere. Loss is identical in
form: `(1 − cos(recon, x)) + β · KL`.

**Why we expected it to help.** CLIP image embeddings are L2-normalized —
they already live on the unit sphere `S^511`, and CLIP similarity is
angular (Wang & Isola, 2020). A Gaussian VAE compresses them into
Euclidean `R^100` and then projects the decoder output back onto the
sphere, an off-manifold detour. The vMF formulation removes that detour:
the latent lives natively on `S^99` and the prior is a meaningful
uniform-on-sphere reference. This was flagged in the paper itself
(Section 5, *Other Untested Alternatives*) as a natural match for CLIP's
manifold but with a caveat — *"hyperspherical VAEs … use spherical
reparameterizations that are less compatible with sign-based
binarization"*.

**Why a previous version of this README was wrong.** A prior commit on
this branch compared *no-binarize sphere* (0.8457) against *binarize
clip* (0.8285) and concluded sphere wins by 1.7pp. That was
apples-to-oranges: the watermarking framework's whole point is the
sign-binarization step, so removing it from one side of the comparison
made the win artificial. The full reproduction below applies the same
binarization pipeline to both models.

**Reproduction (this repo).** Both encoders trained on identical 100-D
latent + cosine recon loss + L2-normalized CLIP-ViT-B/32 features on
the semantic_wm 8k-image dataset, 30 epochs, batch 64, lr=1e-3, Adam +
ReduceLROnPlateau. Two configuration notes:

* **`clip_vae`** uses paper β=0.01 directly.
* **`sphere_vae`** needs **`κ_init=50`** and **β=0** to escape a
  posterior collapse where κ stalls at its softplus floor (≈1, i.e.
  near-uniform vMF) and the encoder outputs a constant. The original
  notebook accidentally avoided this by feeding the encoder
  doubly-resized features (the `Resize((224,224))` → ToTensor →
  ToPILImage → CLIPProcessor path in
  `ablation_sphere_vae.ipynb` cell 14). With the same pipeline our
  `extract_clip_embeddings.py` produces (aspect-preserving single
  bicubic resize), the original β=0.01 setting collapses immediately.
  We provide both extractors so either can be reproduced — see
  `preprocessing/extract_clip_embeddings_notebook_style.py`.

**Pre-binarization reconstruction cosine** (encoder → sample → decoder,
no watermark):

| Model | val CLIP cosine |
|---|---:|
| clip_vae (Gaussian, β=0.01) | **0.8678** |
| sphere_vae (vMF, β=0, κ_init=50) | 0.8392 |

So even **without** sign-binarization, the Gaussian VAE is now the
slightly stronger reconstructor on our setup — the original 0.8457 vs
0.8285 inversion was a preprocessing artifact, not a geometry win.

**With sign-based binarization in the watermarking pipeline** (the
honest, paper-faithful comparison — encoder → 100-bit watermark →
reconstruct → decode → measure cosine vs original CLIP):

| Model | Variant | Overall | Normal | Violence | Sexual |
|---|---|---:|---:|---:|---:|
| **clip_vae** | clip_default (paper pipeline)        | **0.8471** | 0.8134 | 0.8616 | 0.9001 |
| sphere_vae   | clip_default (naive sign of z)        | 0.8219 | 0.7816 | 0.8367 | 0.8876 |
| sphere_vae   | sphere_centered (sign of z − μ_train) | 0.8138 | 0.7727 | 0.8282 | 0.8815 |
| sphere_vae   | sphere_rescaled (bits → ±σ + μ)       | **0.8307** | 0.7931 | 0.8445 | 0.8919 |
| sphere_vae   | sphere_reprojected (rescale + L2)     | 0.8307 | 0.7932 | 0.8446 | 0.8918 |

The Gaussian VAE wins by **~1.6 pp** in the realistic
binarization-included regime even with the strongest sphere recovery
variant (rescale by train-set per-dim std then add train-set mean).
Sphere variants form a clear order: `clip_default` (naive) ≈ baseline,
`sphere_centered` (center first) actually hurts because it discards
the magnitude info that sphere_rescaled then re-injects;
`sphere_reprojected` (rescale, then re-normalize to S^99) gives no
extra over `sphere_rescaled` here.

**Takeaway.** The vMF latent is a clean geometric match for CLIP's
manifold and reconstructs comparably *before* binarization, but inside
the full watermarking pipeline the sphere reparameterization gives up
exactly what the paper's Section 5 predicted: enough information at
the sign-binarize step to fall behind the Gaussian baseline. The
sphere ablation stays in the repo as a latent-quality exploration but
is *not* a drop-in replacement for `clip_vae` in the watermarking
framework.

To reproduce the table above end-to-end:

```bash
# Pre-compute CLIP features the way the notebooks do (one-time, ~10 min CPU)
python preprocessing/extract_clip_embeddings_notebook_style.py
# Train both models on cached features (~70 s each on CPU, faster on GPU)
NUM_EPOCHS=30 python scripts/train_clip_vae.py
NUM_EPOCHS=30 python scripts/train_sphere_vae.py
# Generate the comparison table
python scripts/evaluate_binarization.py
cat results/comparison_table.md
```

---

## Hardware

NVIDIA Tesla T4 (16 GB) — Google Colab free tier. The notebooks unzip
`semantic_wm/dataset.zip` from `/content/drive/MyDrive/semantic_wm/`
into `/content/semantic_wm/` on Colab, and read from `./dataset/`
locally.

---

## License & citation

Code: MIT (see `LICENSE` if present). Datasets are not redistributed.
Citation is suppressed during anonymous review and will be added after
the ICML AI4Good 2026 decision.
