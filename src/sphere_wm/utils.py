"""seed/device helpers + simple checkpoint save/load."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: torch.nn.Module, path: Path, extra: dict | None = None) -> None:
    payload = {"model_state": model.state_dict()}
    if extra is not None:
        payload.update(extra)
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: Path, map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    return payload
