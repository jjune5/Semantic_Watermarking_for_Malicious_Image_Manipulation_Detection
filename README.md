# CLIP-VAE & SDA-Net: Distribution-Based Semantic Watermarking

Anonymous code release accompanying the paper submitted to **ICML AI4Good 2026**.

This repository contains the training and evaluation code for a
distribution-based semantic watermarking framework built on top of
CLIP embeddings, a small VAE, and an SDA-Net (Semantic Distribution
Alignment Network).

> ⚠️ **Sensitive content notice.** Experiments use a class-balanced
> dataset that contains the categories `normal`, `sexual`, and
> `violence`. The image data is **not redistributed** with this
> repository. Researchers must obtain the raw images from the public
> sources listed below and run `preprocessing/prepare_dataset.py` to
> reconstruct the canonical layout.

---

## Repository layout

```
.
├── colab_codes/                  # Notebooks (run in this order — see below)
│   ├── clip_vae.ipynb            # 1. Train CLIP-VAE on the semantic_wm dataset
│   ├── SDA_net.ipynb             # 2. Train SDA-Net on top of the CLIP-VAE features
│   ├── baseline_exp.ipynb        # 3. Baseline comparison (SimHash, classifier, etc.)
│   ├── ablation_beta_vae.ipynb   #    Ablation: KL weight β
│   ├── ablation_vqvae.ipynb      #    Ablation: VQ-VAE in place of VAE
│   ├── ablation_dimensions.ipynb #    Ablation: latent dimensionality
│   ├── ablation_inference.ipynb  #    Ablation: inference-time settings
│   └── ablation_so_lie.ipynb     #    Ablation: SO(n) Lie-algebra rotation on the hyperspherical latent (S^99)
├── preprocessing/
│   └── prepare_dataset.py        # Build dataset/{train,test}/{normal,sexual,violence}/
├── figures/                      # Plots reproduced from notebook outputs
├── requirements.txt
├── .gitignore
└── README.md                     # ← this file
```

---

## Environment setup

Tested on **Python 3.10** with **CUDA 12.1** (Tesla T4 — Google Colab default
configuration).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# External dependency: VINE watermarking (not on PyPI)
git clone https://github.com/Shilin-LU/VINE.git
```

---

## Dataset

The semantic watermarking dataset is built from four public sources:

| Category | Source | License |
|----------|--------|---------|
| `normal` | News Dataset with Images, Kaggle ([link](https://www.kaggle.com/datasets/mdkabinhasan/news-dataset-with-images/data)) | Kaggle terms (check on the dataset page) |
| `violence` | HOD Benchmark Dataset (Ha et al., 2023, WACV-W 2024) — [GitHub](https://github.com/poori-nuna/HOD-Benchmark-Dataset), [arXiv:2310.05192](https://arxiv.org/abs/2310.05192) | **Research-only; redistribution prohibited** |
| `sexual`   | Adult Content Dataset, Figshare ([link](https://figshare.com/articles/dataset/Adult_content_dataset/13456484)) | Figshare release (check on the dataset page) |

### Sample counts and split

| Class | Total | Train | Test |
|-------|-------|-------|------|
| `normal` | 4,000 | 3,200 | 800 |
| `violence` | 2,002 | 1,601 | 401 |
| `sexual` | 2,000 | 1,600 | 400 |

**Train / test split:** 80 / 20 per class, deterministic with
`--seed 42` in `preprocessing/prepare_dataset.py`.

---

## Pre-computed CLIP embeddings (recommended for re-running ablations)

Because the three source datasets have different (and in one case
restrictive) redistribution terms, this repository ships
**pre-computed CLIP-ViT-B/32 image embeddings** instead of the raw
images. The embeddings are 512-D feature vectors that cannot be
inverted back into the source images, but they are sufficient to
re-run every notebook except the original CLIP-VAE training
(`clip_vae.ipynb`), which still needs the raw images to backprop
through the encoder.

```
embeddings/
├── clip_features_train.npy   # (6398, 512) float32   ~12.5 MB
├── clip_features_test.npy    # (1601, 512) float32   ~3.1  MB
├── labels_train.npy          # (6398,)    int64      — 0=normal, 1=violence, 2=sexual
└── labels_test.npy           # (1601,)    int64
```

> Three corrupted JPEG files in the training set were auto-skipped
> by the extractor (6401 → 6398). The extraction script writes a
> matching label array, so the indices stay aligned.

### Producing the embeddings yourself

Once `dataset/` is built (see the previous section), run:

```bash
python preprocessing/extract_clip_embeddings.py \
    --dataset-dir ./dataset \
    --output-dir  ./embeddings \
    --batch-size  32
```

The script auto-selects the best available device
(CUDA → Apple MPS → CPU). On Apple Silicon a full extraction takes a
few minutes; on CPU expect ~15-30 minutes for ~8K images.

### Loading the embeddings

```python
import numpy as np
clip_train = np.load("embeddings/clip_features_train.npy")
clip_test  = np.load("embeddings/clip_features_test.npy")
y_train    = np.load("embeddings/labels_train.npy")
y_test     = np.load("embeddings/labels_test.npy")
```

What you can reproduce from embeddings alone:

| Notebook | Needs raw images? | Needs embeddings? |
|----------|:-----------------:|:------------------:|
| `clip_vae.ipynb` | ✅ yes | — |
| `SDA_net.ipynb` | — | ✅ yes |
| `baseline_exp.ipynb` | partial (watermark visualizations) | ✅ yes |
| `ablation_beta_vae.ipynb`, `ablation_vqvae.ipynb`, `ablation_dimensions.ipynb`, `ablation_inference.ipynb` | — | ✅ yes |
| `ablation_so_lie.ipynb` | ✅ yes (extracts CLIP embeddings on-the-fly during training) | — |

> **Why we do not redistribute the data.**
> The HOD violence dataset explicitly prohibits redistribution
> ("shared for research purposes only … redistribution without
> proper authorization is not permitted"). The adult-content
> dataset and the Kaggle news set are subject to the terms of
> their respective platforms. Each researcher must download the
> three sources directly using the links above; we ship only
> the preprocessing script that arranges them into the canonical
> layout.

Once downloaded, organize the raw images into a directory like:

```
raw_images/
├── normal/    # *.jpg / *.png / *.jpeg / *.bmp
├── sexual/
└── violence/
```

Then run:

```bash
python preprocessing/prepare_dataset.py \
    --input-dir  ./raw_images \
    --output-dir ./dataset \
    --train-ratio 0.8 \
    --resize 512 \
    --seed 42
```

The script verifies each image, optionally resizes to a square
resolution, splits each class with the given ratio, and writes:

```
dataset/
├── train/{normal,sexual,violence}/
└── test/{normal,sexual,violence}/
```

The notebooks expect this exact layout. Per-class counts used in our
experiments are reported in the paper.

---

## Running the notebooks

The notebooks are written for **Google Colab** with a Drive-mounted
dataset zip, but they run unchanged on any Jupyter kernel as long as
the `dataset/` directory above is on the path. Each notebook clears
its outputs after a successful run, so a clean clone of this repo
shows source only.

Suggested execution order:

1. **`colab_codes/clip_vae.ipynb`** — trains the CLIP+VAE (512D → 100D),
   produces the latent representations used by every downstream notebook.
2. **`colab_codes/SDA_net.ipynb`** — trains the Semantic Distribution
   Alignment Network on the CLIP-VAE latents.
3. **`colab_codes/baseline_exp.ipynb`** — full baseline comparison,
   reproduces the main quantitative tables and the F1 ≈ 89% classifier
   reference.
4. The five ablation notebooks
   (`ablation_beta_vae`, `ablation_vqvae`, `ablation_dimensions`,
   `ablation_inference`, `ablation_so_lie`) are independent and can be
   run in any order.  `ablation_so_lie.ipynb` replaces the CLIP-VAE
   latent with an SO(n) Lie-algebra rotation on the hypersphere
   S^99 — see its first cell for the architectural overview.

> If you re-run on Colab, the first cell of each notebook unzips
> `dataset.zip` from `/content/drive/MyDrive/semantic_wm/` into
> `/content/semantic_wm/`. Replace this with a local path if running
> off-Colab.

---

## Hardware and runtime

All experiments were performed on a single **NVIDIA Tesla T4** GPU
(15.83 GB VRAM) — the default Google Colab free-tier GPU.

Approximate wall-clock runtimes per notebook (T4):

| Notebook | Time |
|----------|------|
| `clip_vae.ipynb` | ~25 min |
| `SDA_net.ipynb` | ~15 min |
| `baseline_exp.ipynb` | ~90 min |
| `ablation_beta_vae.ipynb` | ~45 min |
| `ablation_vqvae.ipynb` | ~30 min |
| `ablation_dimensions.ipynb` | ~50 min |
| `ablation_inference.ipynb` | ~20 min |
| `ablation_so_lie.ipynb` | ~45 min |

(Replace the table above with the exact numbers measured for the camera-ready version.)

---

## Reproducing tables and figures

- Quantitative tables in the paper are produced by the cells in
  `baseline_exp.ipynb` and the four ablation notebooks.
- The plots collected under `figures/` correspond directly to the
  figures shown in the paper. File names follow the convention
  `{notebook}__cell{NNN}__img{M}.png`, which matches the cell that
  generated them.

---

## License

Code in this repository is released under the MIT License (see
`LICENSE`, if present). The image dataset is **not** redistributed
under any license — see the dataset section above.

---

## Citation

Anonymous submission. Citation information will be added after review.
