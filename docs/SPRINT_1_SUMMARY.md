# Sprint 1 Summary — Real Data In, Real Labels Out

**Status:** Complete  
**Date:** 2026-04-02

---

## Exit Condition

`python train.py` loads a real 96³ subvolume of raw EM and a real binary mito mask from S3,
prints correct shapes and a non-zero mito voxel count, and saves a side-by-side slice
visualization to `outputs/slice_preview.png`.

All three criteria met.

---

## What Was Implemented

| File | Change |
|------|--------|
| `data/loader.py` | Full S3 zarr loader via `N5FSStore`; percentile normalization for EM; label binarization |
| `train.py` | Steps 1–2 use real data; step 7 visualizes real raw + mask |
| `utils/visualize.py` | Two-panel figure: grayscale EM + RGBA red mito overlay on best-mito Z slice |
| `requirements.txt` | Fixed `fsspec[s3]` quoting (quotes caused pip parse error) |

---

## Bugs Found vs Sprint Plan

### 1. S3 path missing `.n5` suffix
**Plan:** `s3://janelia-cosem-datasets/jrc_hela-2`  
**Actual:** `s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5`  
`zarr.open` raised `PathNotFoundError` until the suffix was added.

### 2. ROI Z coordinate out of bounds
**Plan:** `z=1800` — derived from neuroglancer coordinates using a uniform `nm ÷ 16` divisor  
**Actual:** Z axis max at s2 is 1592; z=1800 overflows  
**Fix:** Z voxel size at s2 is ~21 nm (not 16), so `z_nm ÷ 21 ≈ 1374`. Confirmed mito-dense
regions at runtime by sampling the label array across Z; chose `z=576` (chunk-aligned, 853 mito
voxels in ROI).

### 3. Label values are not binary
**Plan:** `mito_pred/s2` described as binary `{0, 1}`  
**Actual:** Values `{0, 1, 2, …, 9}` observed at runtime  
**Fix:** Binarized in the loader with `(crop > 0).astype(np.uint8)`. The `mask = (labels > 0)`
in `train.py` step 2 was already correct; the fix ensures the loader itself returns clean binary
before any downstream use.

### 4. Visualization: middle Z slice had no mito voxels
**Plan:** Show middle Z slice (z=48 of 96)  
**Actual:** With only 508 mito voxels in the 96³ crop (~5/slice on average), the middle slice
is often empty — the red overlay was invisible.  
**Fix:** `visualize.py` now picks the Z slice with the highest mito voxel count rather than
the fixed middle. Overlay uses RGBA (explicit red + per-pixel alpha) instead of matplotlib's
array-alpha `imshow` parameter, which was silently doing nothing.

---

## Runtime Sanity Check Results

```
raw min/max:         26725, 41441
raw 1st/99th pct:   [27960, 34637]    ← signal clustered as expected, not 0–65535
label unique values: [0 1]             ← binary after binarization
mito voxels:         508               ← non-zero, non-trivial
```

---

## Observations and Carry-Forward

- **Overlay looks sparse:** The red region in `slice_preview.png` is real and spatially
  correct — EM and labels share the same array coordinate system at s2, same ROI, no transform
  needed. Sparsity is genuine: 0.06% foreground at this ROI. `mito_pred` is a model prediction,
  not hand-annotated GT, so visual "meaningfulness" is limited by prediction quality at 16 nm.
  This does **not** indicate an alignment error that would invalidate training.

- **Class imbalance (Sprint 2):** 508 / 884,736 voxels = 0.06% foreground. A naive sampler
  will produce almost entirely background patches. The Sprint 2 patch sampler must oversample
  foreground-containing patches or use weighted loss.

- **zarr v3 migration (backlog):** `N5FSStore` emits a `FutureWarning` about deprecation in
  zarr v3. Pinned `zarr<3` in requirements. Track whether zarr-python v3 provides a clean N5
  replacement before this becomes a blocker.

- **GT label encoding (backlog):** `labels/all` in the publications bucket uses `uint64`
  instance IDs; which ID maps to mitochondria is unknown. Deferred to evaluation sprint.
