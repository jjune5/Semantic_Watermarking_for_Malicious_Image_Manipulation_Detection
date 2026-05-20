"""Loader for pre-computed CLIP features in embeddings/*.npy.

Two feature variants are supported:

* ``feature_set="default"`` (canonical preprocessing — aspect-ratio preserved):
  embeddings/clip_features_{train,test}.npy from
  preprocessing/extract_clip_embeddings.py
* ``feature_set="notebook"`` (matches the reference notebooks bit-for-bit):
  embeddings/clip_features_{train,test}_nb.npy from
  preprocessing/extract_clip_embeddings_notebook_style.py — applies an
  intermediate ``Resize((224, 224))`` + ToTensor/ToPILImage round-trip
  that anisotropically squashes the image to a 1:1 aspect ratio before
  CLIPProcessor sees it. We discovered (the hard way) that this matters
  for HypersphericalVAE training: with the cleaner aspect-preserving
  features the model's κ collapses to its minimum and the encoder
  outputs constant. With the notebook-style features (matching what the
  ablation_sphere_vae.ipynb training run actually saw) sphere_vae
  trains as the original outputs show.

Both variants are L2-normalized at load time. Default ``feature_set``
prefers ``notebook`` when its files exist so the comparison is reproducible.
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np
from torch.utils.data import DataLoader, Dataset

from sphere_wm.config import BATCH_SIZE, EMBEDDINGS_DIR

FeatureSet = Literal["default", "notebook", "auto"]


def _feature_paths(split: str, feature_set: FeatureSet):
    base = "clip_features_" + split
    nb = (EMBEDDINGS_DIR / f"{base}_nb.npy", EMBEDDINGS_DIR / f"labels_{split}_nb.npy")
    default = (EMBEDDINGS_DIR / f"{base}.npy", EMBEDDINGS_DIR / f"labels_{split}.npy")
    if feature_set == "notebook":
        return nb
    if feature_set == "default":
        return default
    # auto — prefer notebook when present
    if nb[0].exists() and nb[1].exists():
        return nb
    return default


class CLIPFeatureDataset(Dataset):
    def __init__(
        self,
        split: str,
        normalize: bool = True,
        feature_set: FeatureSet = "auto",
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        feats_path, labels_path = _feature_paths(split, feature_set)
        if not feats_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"missing {feats_path} or {labels_path}; "
                f"run preprocessing/extract_clip_embeddings*.py first"
            )
        feats = np.load(feats_path).astype(np.float32)
        if normalize:
            n = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / np.clip(n, 1e-12, None)
        self.features = feats
        self.labels = np.load(labels_path).astype(np.int64)
        self.feature_set_used = "notebook" if "_nb.npy" in feats_path.name else "default"

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        return self.features[i], int(self.labels[i])


def make_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    feature_set: FeatureSet = "auto",
) -> tuple[DataLoader, DataLoader]:
    fset = os.environ.get("FEATURE_SET", feature_set)
    train = CLIPFeatureDataset("train", feature_set=fset)
    test = CLIPFeatureDataset("test", feature_set=fset)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=True),
        DataLoader(test, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )
