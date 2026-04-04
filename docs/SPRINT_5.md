# Sprint 5 — Inference and Visualization

**Status:** Ready for implementation.

**Goal:** Load the trained checkpoint, run inference on the held-out region, and produce
a side-by-side visualization of the predicted mask vs ground truth. This completes the MVP.

**Exit condition:** `predict.py` loads `checkpoints/epoch_030.pt`, runs the model on
`HOLDOUT_ROI`, and saves `outputs/inference_preview.png` showing three panels: raw EM,
ground truth mask, and predicted mask. The predicted mask must show non-trivial structure —
not all zeros and not all ones.

---

## Pre-Sprint Checklist

- [ ] Comment out `fg in center crop` debug prints in `data/sampler.py`
- [ ] Confirm `checkpoints/epoch_030.pt` exists and is non-empty
- [ ] Confirm `HOLDOUT_ROI = (slice(800, 932), slice(80, 212), slice(1200, 1332))`
  is defined in `train.py` and was never sampled during training

---

## Context

| Item | Value |
|------|-------|
| Checkpoint | `checkpoints/epoch_030.pt` |
| Holdout ROI | `Z: 800–932, Y: 80–212, X: 1200–1332` (132³ voxels) |
| Model input | `(1, 1, 132, 132, 132)` float32, percentile-normalized |
| Model output | `(1, 1, 40, 40, 40)` raw logits — apply sigmoid to get probabilities |
| Threshold | 0.5 — sigmoid output > 0.5 = predicted foreground |
| Ground truth | `labels/mito_seg/s2` at same ROI, binarized with `> 0` |
| Training loss | Epoch 1: 0.756 → Epoch 30: 0.481 (35% reduction) |

---

## Tasks

### Step 1 — Create `predict.py`

New standalone script — does not modify `train.py`.

```python
"""
predict.py — Sprint 5 inference script.
Loads trained checkpoint, runs inference on held-out region,
saves side-by-side visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import zarr

from data.loader import open_arrays, PATCH_SIZE
from model.unet import get_model

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = "checkpoints/epoch_030.pt"
OUTPUT_PATH = "outputs/inference_preview.png"

# Held-out region — never seen during training
HOLDOUT_ROI = (slice(800, 932), slice(80, 212), slice(1200, 1332))
THRESHOLD   = 0.5
```

### Step 2 — Load checkpoint and run inference

```python
# Load model
model = get_model()
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.eval()

# Load data
em_array, seg_array = open_arrays()

# Load holdout crop
raw_crop = em_array[HOLDOUT_ROI].astype(np.float32)
p1, p99  = np.percentile(raw_crop, [1, 99])
raw_norm = np.clip((raw_crop - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

seg_crop   = seg_array[HOLDOUT_ROI]
gt_binary  = (seg_crop > 0).astype(np.uint8)

# Forward pass — no gradients needed
x = torch.from_numpy(raw_norm[np.newaxis, np.newaxis])  # (1, 1, 132, 132, 132)
with torch.no_grad():
    logits = model(x)                                    # (1, 1, 40, 40, 40)

probs = torch.sigmoid(logits).squeeze().numpy()          # (40, 40, 40)
pred  = (probs > THRESHOLD).astype(np.uint8)             # (40, 40, 40)
```

### Step 3 — Align ground truth to output shape

The model output is 40³, center-cropped from the 132³ input. The ground truth needs
to be cropped to the same 40³ region for a fair comparison:

```python
from train import center_crop  # reuse existing helper

gt_tensor    = torch.from_numpy(gt_binary[np.newaxis, np.newaxis].astype(np.float32))
gt_cropped   = center_crop(gt_tensor, (40, 40, 40)).squeeze().numpy().astype(np.uint8)

# Also crop raw EM for display
raw_tensor   = torch.from_numpy(raw_norm[np.newaxis, np.newaxis])
raw_cropped  = center_crop(raw_tensor, (40, 40, 40)).squeeze().numpy()
```

### Step 4 — Visualize

Pick the Z slice with the most ground truth foreground for display:

```python
best_z = np.argmax(gt_cropped.sum(axis=(1, 2)))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Raw EM
axes[0].imshow(raw_cropped[best_z], cmap="gray")
axes[0].set_title(f"Raw EM — z={best_z}")
axes[0].axis("off")

# Panel 2: Ground truth
axes[1].imshow(raw_cropped[best_z], cmap="gray")
rgba_gt = np.zeros((*gt_cropped[best_z].shape, 4))
rgba_gt[gt_cropped[best_z] > 0] = [0, 1, 0, 0.5]   # green = ground truth
axes[1].imshow(rgba_gt)
axes[1].set_title(f"Ground truth — z={best_z}\n({gt_cropped[best_z].sum()} fg voxels)")
axes[1].axis("off")

# Panel 3: Prediction
axes[2].imshow(raw_cropped[best_z], cmap="gray")
rgba_pred = np.zeros((*pred[best_z].shape, 4))
rgba_pred[pred[best_z] > 0] = [1, 0, 0, 0.5]        # red = prediction
axes[2].imshow(rgba_pred)
fg_pct = 100 * pred[best_z].mean()
axes[2].set_title(f"Prediction (thresh={THRESHOLD}) — z={best_z}\n({pred[best_z].sum()} fg voxels, {fg_pct:.1f}%)")
axes[2].axis("off")

plt.suptitle(f"Inference on held-out region | Checkpoint: {CHECKPOINT}", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved: {OUTPUT_PATH}")
```

### Step 5 — Print inference statistics

```python
print(f"\nInference statistics:")
print(f"  Holdout ROI:        {HOLDOUT_ROI}")
print(f"  Output shape:       {pred.shape}")
print(f"  GT fg voxels:       {gt_cropped.sum()} / {gt_cropped.size} ({100*gt_cropped.mean():.2f}%)")
print(f"  Pred fg voxels:     {pred.sum()} / {pred.size} ({100*pred.mean():.2f}%)")
print(f"  Logit min/max:      {logits.min():.4f} / {logits.max():.4f}")
print(f"  Prob min/max:       {probs.min():.4f} / {probs.max():.4f}")
```

### Step 6 — Sanity checks

- [ ] `outputs/inference_preview.png` saved
- [ ] Logit min/max is not `0.0000 / 0.0000` — model is not outputting flat zero
- [ ] Prob min/max shows a range (not all 0.5000) — sigmoid is working
- [ ] Predicted fg voxel count is between 1% and 99% of total — not all zeros, not all ones
- [ ] Visualization committed to repo — this is the MVP artifact

---

## Commit Plan

| Commit | Message |
|--------|---------|
| 1 | `sprint5: remove sampler debug prints` |
| 2 | `sprint5: implement predict.py with inference and visualization` |
| 3 | `sprint5: inference_preview.png committed — MVP complete` |

---

## Notes

- **`model.eval()` is critical.** Without it, any dropout or BatchNorm layers (if present)
  behave differently at inference. This model has no BatchNorm, but `eval()` is correct
  practice regardless.

- **`torch.no_grad()` reduces memory and speeds up inference.** No gradients are needed
  for a forward pass without training.

- **Ground truth color is green, prediction is red.** Where they overlap will appear
  yellow-ish. Perfect predictions would show yellow everywhere there's signal.

- **30 epochs with 240 gradient steps is a minimal training run.** The prediction will
  not be perfect — some mitochondria will be missed, some false positives expected.
  The goal is non-trivial structure in the prediction, not accuracy.

- **196-minute training time and growing backward pass (backlog).** The backward pass
  grew from ~174s at epoch 1 to ~940s at epoch 25. Root cause unknown — possible gradient
  graph accumulation. Must be investigated before any extended training runs.
  Added to `TODO_ENHANCEMENTS.md`.
