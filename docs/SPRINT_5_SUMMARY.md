# Sprint 5 Summary — Inference and Visualization

**Status:** Complete — MVP achieved  
**Date:** 2026-04-03

> All figures taken from actual console output. No values estimated or inferred.

---

## Exit Condition

`predict.py` loads `checkpoints/epoch_030.pt`, runs the model on a held-out region,
and saves `outputs/inference_preview_contrast_stretched.png` showing four panels: raw EM,
ground truth mask, thresholded prediction, and contrast-stretched probability map.
The probability map shows non-trivial spatial structure confirming the pipeline is functional.

**Met.**

---

## What Was Implemented

| File | Change |
|------|--------|
| `predict.py` | New script: loads checkpoint, runs inference on `HOLDOUT_ROI`, saves 4-panel visualization |
| `train.py` | `HOLDOUT_ROI` updated to verified foreground-containing region |

---

## Inference Statistics — Actual Console Output

```
Holdout ROI:     (slice(200, 332), slice(80, 212), slice(1500, 1632))
Output shape:    (40, 40, 40)
GT fg voxels:    6135 / 64000 (9.59%)
Pred fg voxels:  44710 / 64000 (69.86%)  [threshold=0.3]
Logit min/max:   -1.0167 / -0.7356
Prob min/max:    0.2657 / 0.3240
```

---

## Bugs Found

### 1. Original `HOLDOUT_ROI` had zero foreground
The ROI committed in Sprint 4.1 (`Z=800–932, X=1200–1332`) had 0 foreground voxels in
`mito_seg/s2`. The Sprint 4.1 plan included a verification step that was not executed.
Fixed by running `inspect_holdout.py` to survey candidate regions and selecting
`Z=200–332, X=1500–1632` (fg=10.4%, 239K foreground voxels across the full 132³ crop).

### 2. Input tensor dtype mismatch (float64 vs float32)
`predict.py` raised `RuntimeError: Input type (double) and bias type (float) should be
the same`. `np.percentile` returns `float64` which propagated into the tensor despite
`.astype(np.float32)` on the array. Fixed by adding `.float()` to the tensor construction:
```python
x = torch.from_numpy(raw_norm[np.newaxis, np.newaxis]).float()
```

---

## Visualization

Four-panel output at `outputs/inference_preview_contrast_stretched.png`:

| Panel | Contents |
|-------|----------|
| Raw EM | Grayscale FIB-SEM at best Z slice (z=22 of 40) |
| Ground truth | Green overlay — `mito_seg/s2` binarized, 412 fg voxels at this Z |
| Prediction | Red overlay at threshold=0.3 — 1394 fg voxels (87.1%) — over-predicts |
| Prob map | Contrast-stretched probability heatmap — shows spatial structure |

The thresholded prediction panel over-predicts due to the model's narrow probability
range (0.2657–0.3240). Any threshold in this range is effectively arbitrary. The
contrast-stretched probability map is the honest representation of what was learned.

---

## Model Quality Assessment

The model produces a probability field with spatial structure but insufficient
discrimination. This is expected given the training budget:

| Factor | Value | Impact |
|--------|-------|--------|
| Gradient steps | 240 (30 epochs × 8 batches) | Far too few for 93M params |
| Model parameters | 93,633,689 | Large — would benefit from GPU |
| Prob range | 0.2657–0.3240 | 0.058 span — no meaningful threshold exists |
| Loss reduction | 0.756 → 0.481 (epoch 1→30) | Real learning signal, insufficient convergence |
| Training time | 196.6 minutes (CPU) | Extended training requires GPU |

The pipeline is correct. The model requires extended training to produce useful predictions.

---

## MVP Status

All five pipeline components are implemented and functional:

| Component | Status |
|-----------|--------|
| S3 data loading (zarr/N5) | ✅ Complete |
| Binary mask from `mito_seg/s2` | ✅ Complete |
| Foreground-biased patch sampler | ✅ Complete |
| funkelab U-Net training with BCE loss | ✅ Complete |
| Inference + visualization | ✅ Complete |

---

## Carry-Forward

- **Extended training (highest priority):** GPU support + more epochs is the single most
  impactful next step. See `TODO_ENHANCEMENTS.md`.

- **Growing backward pass time:** Epoch 25 took 940s. Root cause unidentified. Must be
  resolved before any run longer than 30 epochs. See `TODO_ENHANCEMENTS.md`.

- **Sampler debug prints:** `fg in center crop:` still printing per attempt. Remove from
  `data/sampler.py` before sharing the repo.

- **README + cover letter:** Add `inference_preview_contrast_stretched.png` to README.
  Update cover letter with GitHub link.

- **Email Jan:** Share repo and describe findings — activation bug, gradient explosion,
  background collapse, and what extended training would require.
