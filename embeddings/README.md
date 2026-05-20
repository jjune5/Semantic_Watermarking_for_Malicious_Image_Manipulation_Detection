# Pre-computed CLIP embeddings

Shipped feature files (CLIP-ViT-B/32, 512-D image embeddings):

| File | Shape | Dtype | Size |
|------|-------|-------|------|
| `clip_features_train.npy` | (6398, 512) | float32 | ~12.5 MB |
| `clip_features_test.npy`  | (1601, 512) | float32 | ~3.1 MB |
| `labels_train.npy` | (6398,) | int64 | 50 KB |
| `labels_test.npy`  | (1601,) | int64 | 13 KB |

**Label mapping:** `0 = normal`, `1 = violence`, `2 = sexual`.

Three corrupted JPEG files in the train split were skipped during
extraction (6401 → 6398). The accompanying label array stays in sync.

To regenerate from scratch:

```bash
python preprocessing/extract_clip_embeddings.py \
    --dataset-dir ./dataset \
    --output-dir  ./embeddings \
    --batch-size  32
```

Loading example:

```python
import numpy as np
X_train = np.load("embeddings/clip_features_train.npy")  # (6398, 512)
y_train = np.load("embeddings/labels_train.npy")         # (6398,)
```
