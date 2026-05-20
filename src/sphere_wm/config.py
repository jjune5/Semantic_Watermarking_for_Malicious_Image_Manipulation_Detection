"""Hyperparameters and paths shared across training + evaluation scripts.

All values can be overridden via environment variables for one-off sweeps
without editing this file. Resolved at import time."""
from __future__ import annotations

import os
from pathlib import Path

import torch


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "semantic_wm" / "dataset"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
for d in (CHECKPOINT_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True, parents=True)

INPUT_DIM = 512
HIDDEN_DIM = 256
LATENT_DIM = 100
NUM_EPOCHS = _env_int("NUM_EPOCHS", 50)
BATCH_SIZE = _env_int("BATCH_SIZE", 64)
LEARNING_RATE = _env_float("LR", 1e-3)
BETA = _env_float("BETA", 0.01)
SEED = _env_int("SEED", 42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORIES = ["normal", "violence", "sexual"]
