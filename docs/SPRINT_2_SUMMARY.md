# Sprint 2 Summary — Patch Sampler

**Status:** Complete  
**Date:** 2026-04-03

---

## Exit Condition

`python train.py` opens a lazy handle to the full `mito_seg/s2` volume, samples 8 real
patches with foreground-biased rejection sampling, logs mean foreground fraction per batch,
and passes correctly shaped tensors to the stub model. The slice preview shows the red
overlay falling on the bright-outlined oval mitochondria structures in the EM.

All criteria met. Mean fg_frac: 0.11. Best patch in batch: 19.7%.

---

## What Was Implemented

| File | Change |
|------|--------|
| `data/loader.py` | `SEG_LAYER` → `labels/mito_seg/s2`; `N5_PATH` constant made explicit; `PATCH_SIZE=132` added; ROI updated to inspection-confirmed origin (480, 80, 2382); `open_arrays()` added for lazy full-volume zarr handle access |
| `data/sampler.py` | Full rewrite — foreground-biased rejection sampler; accepts zarr handles; percentile-normalizes raw crops; returns `(raw, label, fg_frac)` triples; warns on `max_attempts` exhaustion |
| `train.py` | Steps 1–3 updated: opens zarr handles, defers mask generation to sampler, logs mean fg_frac; step 7 passes best-fg-fraction patch to visualizer |
| `utils/visualize.py` | Uses middle Z slice of the provided patch; accepts `fg_frac` parameter shown in plot titles |

---

## Shape Test Results

| Input | Output | Valid? |
|-------|--------|--------|
| 64³ | FAILED | ✗ — too small for 3× downsampling |
| 96³ | FAILED | ✗ — too small for 3× downsampling |
| 132³ | (1, 1, 40, 40, 40) | ✓ — selected |
| 148³ | (1, 1, 56, 56, 56) | ✓ — reserve option |

`PATCH_SIZE = 132`. Y dimension (400 voxels) leaves valid origins in `[0, 268]` — no constraint.

---

## Bugs Found vs Sprint Plan

### 1. System Python / zarr version conflict
**Context:** `pip install funlib.learn.torch` was run in the system Python (3.14), which
upgraded zarr to v3.1.6. zarr v3 removed `N5FSStore`, breaking the pipeline immediately.  
**Fix:** All commands must be run in the project venv (`.venv/`, Python 3.12, zarr 2.18.7).
The system Python 3.14 cannot downgrade to zarr v2 because `numcodecs` has no 3.14 wheel and
fails to build from source.  
**Carry-forward:** The zarr v3 migration warning on `N5FSStore` is now more urgent — pinned
`zarr<3` keeps the venv stable but this will need a proper fix (likely `n5py` or zarr v3
N5 compatibility layer) before Python 3.14 can be the runtime.

---

## Runtime Sanity Check Results

```
[1/7] Loading subvolume...        shape: (1592, 400, 3000)
[2/7] Generating binary mask...   deferred to sampler
[3/7] Sampling patches...         n=8, fg_frac: 0.11
[4/7] Instantiating model...      params: 0 (stub)
[5/7] Forward pass...             output shape: (1, 1, 132, 132, 132)
[6/7] Computing loss...           loss: 0.8513 (stub)
[7/7] Saving visualization...     outputs/slice_preview.png
```

Visual check: red overlay falls on bright-outlined oval structures, not on extracellular
space. This confirms `mito_seg/s2` is correctly aligned and correctly binarized.

---

## Observations and Carry-Forward

- **`mito_pred/s2` rejected:** Pre-sprint inspection confirmed this layer labels extracellular
  space, not mitochondria, at the Sprint 1 ROI. `mito_seg/s2` (instance segmentation IDs 47,
  110, 138, …) is the correct label source. The visual sanity check is essential — nonzero
  voxel counts alone do not prove the overlay is on the right structure.

- **Foreground coverage looks healthy:** 6.79% foreground across the full `mito_seg/s2` volume
  (from pre-sprint inspection). The sampler achieves 11% mean fg_frac with `min_fg_frac=0.05`,
  indicating the rejection loop is working and the volume has good coverage. If Sprint 3
  training shows slow convergence, revisit rejection rate logging first.

- **Sprint 3 model wiring:** funkelab U-Net is already installed (`funlib.learn.torch 0.1.0`).
  The identity stub currently passes 132³ through unchanged. Replacing it with the real U-Net
  will produce 40³ outputs — the loss target must be cropped or the model output upsampled to
  match. The center-crop approach (crop the 132³ label to 40³ around the center) is standard
  for valid-convolution U-Nets.

- **`PATCH_SIZE` as single source of truth:** Currently defined in `data/loader.py` and
  imported into `train.py`. Sprint 4 config file should consolidate this along with
  `N_PATCHES`, `OUTPUT_DIR`, and any loss weights.

- **zarr v3 / N5 (backlog):** `N5FSStore` deprecation warning is present in every run.
  Investigate `n5py` as the zarr v3-compatible replacement before upgrading the Python runtime.
