"""Loader for pre-computed CLIP features in embeddings/*.npy.

Layout (produced by preprocessing/extract_clip_embeddings.py):

    embeddings/
    ├── clip_features_train.npy   (N_train, 512) float32
    ├── clip_features_test.npy    (N_test,  512) float32
    ├── labels_train.npy          (N_train,)     int64
    └── labels_test.npy           (N_test,)      int64

Features are L2-normalized at load time so every downstream module sees
CLIP embeddings on the unit sphere (matching how `F.cosine_similarity`
is used as the reconstruction loss).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sphere_wm.config import BATCH_SIZE, EMBEDDINGS_DIR


class CLIPFeatureDataset(Dataset):
    def __init__(self, split: str, normalize: bool = True):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        feats_path = EMBEDDINGS_DIR / f"clip_features_{split}.npy"
        labels_path = EMBEDDINGS_DIR / f"labels_{split}.npy"
        if not feats_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"missing {feats_path} or {labels_path}; "
                f"run preprocessing/extract_clip_embeddings.py first"
            )
        feats = np.load(feats_path).astype(np.float32)
        if normalize:
            n = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / np.clip(n, 1e-12, None)
        self.features = feats
        self.labels = np.load(labels_path).astype(np.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        return self.features[i], int(self.labels[i])


def make_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train = CLIPFeatureDataset("train")
    test = CLIPFeatureDataset("test")
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=True),
        DataLoader(test, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )
