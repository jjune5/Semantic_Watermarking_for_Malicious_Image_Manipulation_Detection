"""
Extract CLIP-ViT-B/32 image embeddings from a prepared dataset and write
them as four .npy files that the SDA-Net and ablation notebooks can load
directly.

Input layout (produced by `prepare_dataset.py`):

    dataset/
    ├── train/{normal,sexual,violence}/*.{jpg,png,jpeg,bmp}
    └── test/ {normal,sexual,violence}/*.{jpg,png,jpeg,bmp}

Output layout:

    embeddings/
    ├── clip_features_train.npy   (N_train, 512) float32
    ├── clip_features_test.npy    (N_test,  512) float32
    ├── labels_train.npy          (N_train,)     int   (0=normal, 1=violence, 2=sexual)
    └── labels_test.npy           (N_test,)      int

Device selection is automatic: CUDA → Apple MPS → CPU. The full pipeline
runs in a few minutes on a single GPU and ~15-30 minutes on CPU for the
~8K images used in this paper.

Usage:

    python preprocessing/extract_clip_embeddings.py \
        --dataset-dir ./dataset \
        --output-dir  ./embeddings \
        --batch-size  32
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CATEGORIES = ("normal", "violence", "sexual")
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CATEGORIES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_split(split_dir: Path) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for cat in CATEGORIES:
        class_dir = split_dir / cat
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        cat_files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        paths.extend(cat_files)
        labels.extend([LABEL_TO_INDEX[cat]] * len(cat_files))
    return paths, labels


def load_image(path: Path) -> Image.Image | None:
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"  [skip] {path.name}: {e}")
        return None


def encode_batch(
    images: list[Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    return feats.detach().cpu().numpy().astype(np.float32)


def extract_split(
    split_dir: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    paths, labels = collect_split(split_dir)
    n = len(paths)
    feats = np.zeros((n, 512), dtype=np.float32)
    keep_mask = np.zeros(n, dtype=bool)

    pbar = tqdm(range(0, n, batch_size), desc=f"{split_dir.name:5s}", unit="batch")
    for start in pbar:
        chunk_paths = paths[start : start + batch_size]
        loaded = [(i + start, load_image(p)) for i, p in enumerate(chunk_paths)]
        valid = [(idx, img) for idx, img in loaded if img is not None]
        if not valid:
            continue
        idxs, imgs = zip(*valid)
        batch_feats = encode_batch(list(imgs), model, processor, device)
        feats[list(idxs)] = batch_feats
        for i in idxs:
            keep_mask[i] = True

    if keep_mask.sum() < n:
        print(f"  Dropped {n - keep_mask.sum()} unreadable image(s) from {split_dir.name}")
    feats = feats[keep_mask]
    label_arr = np.asarray(labels, dtype=np.int64)[keep_mask]
    return feats, label_arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"),
                        help="Directory containing train/ and test/ (default: ./dataset).")
    parser.add_argument("--output-dir", type=Path, default=Path("embeddings"),
                        help="Where to write the four .npy files (default: ./embeddings).")
    parser.add_argument("--batch-size", type=int, default=32, help="Forward batch size (default: 32).")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device      : {device}")
    print(f"Dataset dir : {args.dataset_dir.resolve()}")
    print(f"Output dir  : {args.output_dir.resolve()}")
    print(f"Batch size  : {args.batch_size}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {CLIP_MODEL_NAME} ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    print("Model loaded.\n")

    for split in ("train", "test"):
        split_dir = args.dataset_dir / split
        feats, labels = extract_split(split_dir, model, processor, device, args.batch_size)
        np.save(args.output_dir / f"clip_features_{split}.npy", feats)
        np.save(args.output_dir / f"labels_{split}.npy", labels)
        print(f"  {split}: features={feats.shape}, labels={labels.shape}\n")

    print(f"Done. Files written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
