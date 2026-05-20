"""
Prepare the semantic_wm dataset directory structure used by all notebooks.

The image dataset is not redistributed. Researchers must obtain the raw images
from their original public sources (see README) and place them under a single
input directory, organized by class. This script then:

  1. Verifies that each image can be opened (skips corrupted files)
  2. (Optional) Resizes images to a fixed resolution
  3. Splits each class into train/test with a configurable ratio
  4. Writes the canonical dataset/ structure expected by the notebooks:

        dataset/
        ├── train/
        │   ├── normal/
        │   ├── sexual/
        │   └── violence/
        └── test/
            ├── normal/
            ├── sexual/
            └── violence/

Usage:

    python preprocessing/prepare_dataset.py \
        --input-dir  ./raw_images \
        --output-dir ./dataset \
        --train-ratio 0.8 \
        --resize 512 \
        --seed 42

The expected layout of --input-dir is:

    raw_images/
    ├── normal/    *.jpg|*.png|*.jpeg|*.bmp
    ├── sexual/
    └── violence/
"""

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

CATEGORIES = ("normal", "sexual", "violence")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def open_or_skip(path: Path) -> Image.Image | None:
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"  [skip] {path.name}: {e}")
        return None


def write_image(img: Image.Image, dst: Path, resize: int | None) -> None:
    if resize is not None:
        img = img.resize((resize, resize), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=95)


def split_class(
    class_name: str,
    src_dir: Path,
    out_root: Path,
    train_ratio: float,
    resize: int | None,
    rng: random.Random,
) -> dict[str, int]:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Missing input class directory: {src_dir}")

    paths = collect_images(src_dir)
    rng.shuffle(paths)
    cut = int(len(paths) * train_ratio)
    train_paths, test_paths = paths[:cut], paths[cut:]

    counts = {"train": 0, "test": 0, "skipped": 0}
    for split, paths_in_split in (("train", train_paths), ("test", test_paths)):
        for src in paths_in_split:
            img = open_or_skip(src)
            if img is None:
                counts["skipped"] += 1
                continue
            dst = out_root / split / class_name / src.name
            write_image(img, dst, resize)
            counts[split] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing one subfolder per class (normal/sexual/violence).")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"),
                        help="Output dataset root (default: ./dataset).")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of each class assigned to train (default: 0.8).")
    parser.add_argument("--resize", type=int, default=None,
                        help="If set, resize all images to this square resolution (e.g. 512).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffling seed (default: 42).")
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be in (0, 1).")

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input : {args.input_dir.resolve()}")
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Split : train={args.train_ratio:.2f}, test={1 - args.train_ratio:.2f}")
    print(f"Resize: {args.resize or 'none'}\n")

    grand_total = {"train": 0, "test": 0, "skipped": 0}
    for cat in CATEGORIES:
        print(f"[{cat}]")
        counts = split_class(cat, args.input_dir / cat, args.output_dir,
                             args.train_ratio, args.resize, rng)
        for k, v in counts.items():
            grand_total[k] += v
        print(f"  train={counts['train']}, test={counts['test']}, "
              f"skipped={counts['skipped']}\n")

    print("=" * 50)
    print(f"Total: train={grand_total['train']}, test={grand_total['test']}, "
          f"skipped={grand_total['skipped']}")
    print(f"Done. Output written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
