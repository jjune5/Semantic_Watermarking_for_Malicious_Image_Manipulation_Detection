"""Extract CLIP-ViT-B/32 features the SAME way the reference notebooks do.

Pipeline (matches ablation_sphere_vae.ipynb cell 11+14 and clip_vae.ipynb):
  PIL.open(path).convert('RGB')
    → transforms.Resize((224, 224))     # squash aspect ratio (bilinear)
    → transforms.ToTensor()
    → transforms.ToPILImage()           # back to PIL ([0,1] float → uint8)
    → CLIPProcessor                     # CLIP's own preprocessing (bicubic resize/center crop, normalize)
    → CLIPModel.vision_model(...)[1]    # pooled output (768D)
    → CLIPModel.visual_projection(...)  # (512D)
    → F.normalize(p=2, dim=-1, eps=1e-8)

The intermediate ``Resize((224,224))`` differs from the more natural "PIL →
CLIPProcessor" path because CLIPProcessor preserves aspect ratio via
shortest-edge resize + center crop, while ``Resize((224,224))`` does an
anisotropic squash. We discovered the difference matters for sphere_vae
training (anisotropic features carry enough additional structure to keep
vMF κ from collapsing to its minimum). To stay faithful to the notebook,
we replicate the double-resize here. Outputs go to embeddings/*_nb.npy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "semantic_wm" / "dataset"
OUT = REPO / "embeddings"
OUT.mkdir(exist_ok=True)
CATEGORIES = ["normal", "violence", "sexual"]
LABEL_OF = {c: i for i, c in enumerate(CATEGORIES)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    squash = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    to_pil = transforms.ToPILImage()

    for split in ("train", "test"):
        items: list[tuple[Path, int]] = []
        for cat in CATEGORIES:
            for p in sorted((DATASET / split / cat).glob("*.jpg")):
                items.append((p, LABEL_OF[cat]))
        print(f"{split}: {len(items)} images")
        feats, labels = [], []
        batch_size = 64
        for start in tqdm(range(0, len(items), batch_size), desc=split):
            chunk = items[start : start + batch_size]
            imgs, lbls = [], []
            for p, y in chunk:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(to_pil(squash(img)))
                    lbls.append(y)
                except Exception as e:
                    print(f"skip {p.name}: {e}")
            if not imgs:
                continue
            inputs = proc(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                v = clip.vision_model(**inputs)[1]
                f = clip.visual_projection(v)
                f = F.normalize(f, p=2, dim=-1, eps=1e-8)
            feats.append(f.cpu().numpy().astype(np.float32))
            labels.extend(lbls)
        feats = np.concatenate(feats, axis=0)
        labels = np.asarray(labels, dtype=np.int64)
        np.save(OUT / f"clip_features_{split}_nb.npy", feats)
        np.save(OUT / f"labels_{split}_nb.npy", labels)
        print(f"  saved {feats.shape} → embeddings/clip_features_{split}_nb.npy")


if __name__ == "__main__":
    main()
