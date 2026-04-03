# Pre-Sprint 2 Inspection Findings

**Date:** 2026-04-03  
**Scripts:** `inspect_binarization.py`, `inspect_roi.py`  
**Purpose:** Determine correct binarization strategy and ROI origin before writing the Sprint 2 patch sampler.

---

## 1. `mito_pred` Value Encoding

### Finding: `mito_pred/s2` is a multi-class label array, not a soft prediction

| Layer | dtype | Unique values | Interpretation |
|-------|-------|---------------|----------------|
| `mito_pred/s0` (sampled crop) | uint8 | `{0}` | All background — sampled crop is outside any labeled region |
| `mito_pred/s2` (Sprint 1 ROI) | uint8 | `{0, 1, 2, 3, 4, 5, 6, 7}` | Multi-class instance or class IDs, **not** soft predictions |

The heinrich-2021a README describes predictions as "a scalar field of uint8 values HIGH inside an object class and LOW outside," implying 0–255 soft scores. The actual values `{0–7}` at s2 contradict a soft-score interpretation — values never exceed 12 across the broader survey region. These are class/instance IDs introduced by downsampled segmentation, not averaging artifacts.

### Binarization conclusion: `label > 0`

Any nonzero value is foreground. The Sprint 1 loader's `(crop > 0).astype(np.uint8)` is correct. Thresholds ≥ 10 produce zero foreground voxels.

---

## 2. `mito_pred` vs `mito_seg` Overlap

| Metric | Value |
|--------|-------|
| `pred > 0` AND `seg > 0` voxels | **0** |
| Only `pred > 0` voxels | 4,683 |
| Only `seg > 0` voxels | 94,155 |

Zero overlap between `mito_pred/s2` and `mito_seg/s2` at the Sprint 1 ROI. These two layers label spatially non-overlapping regions at this location, or represent different object classes. `mito_seg` (uint16 with instance IDs up to 388) covers ~5.3% of voxels vs `mito_pred`'s 0.26%. 

**Sprint 2 implication:** Do not combine or cross-validate `mito_pred` and `mito_seg` as equivalent ground-truth sources without further investigation.

---

## 3. ROI Density and Origin

### Survey region
`Z: slice(400, 800), Y: slice(80, 200), X: slice(2350, 2510)` — 400 × 120 × 160 voxels at s2.

### Label density by Z (top 10)

| Z slice | Mito voxels | % of XY plane |
|---------|-------------|---------------|
| 575 | 2,187 | 11.39% |
| 574 | 1,965 | 10.23% |
| 576 | 1,477 | 7.69% |
| 577 | 1,428 | 7.44% |
| 573 | 1,309 | 6.82% |
| 572 | 1,261 | 6.57% |
| 570 | 1,214 | 6.32% |
| 569 | 1,191 | 6.20% |
| 568 | 1,128 | 5.88% |
| 567 | 1,113 | 5.80% |

Density is concentrated in a tight band **z = 566–578**, with a secondary cluster around z = 621–627. Sprint 1's origin at z=576 was near-optimal (3rd densest slice).

### Recommended ROI for Sprint 2

```python
ROI_Z = slice(480, 576)   # chunk-aligned, covers the dense z=566-575 band
ROI_Y = slice(80,  176)   # 96 voxels
ROI_X = slice(2382, 2478) # 96 voxels, centered on survey X range
```

Verify visually in `outputs/inspect_roi_density.png` (Z density profile + best-slice overlay) and `outputs/inspect_roi_grid.png` (3×3 grid of slices around z=575).

---

## 4. Visual Outputs

| File | Contents |
|------|----------|
| `outputs/inspect_roi_density.png` | Z density bar chart, EM+overlay at best Z (z=575), EM+overlay at Sprint 1 Z (z=576) |
| `outputs/inspect_roi_grid.png` | 3×3 grid of Z slices ± 4 slices around z=575, each with mito_pred overlay |

---

## 5. Open Questions for Sprint 2

1. **`mito_pred/s0` is all zeros at the sampled crop.** The s0 crop used (`z: 2300–2400, y: 360–456, x: 9560–9656`) may not correspond to the same physical location as the s2 dense region. The s2→s0 coordinate scaling was approximate. A corrected s0 crop should be checked before using s0-resolution labels.

2. **Zero overlap between `mito_pred` and `mito_seg`.** This warrants investigation — either the two layers annotate different cellular structures, or there is a coordinate system mismatch between them that has not yet been diagnosed.

3. **Class imbalance persists.** Even at the densest Z slice (z=575), foreground is 11.4% of the XY plane. Over a full 96³ patch, foreground will be sparse. Sprint 2's patch sampler should oversample foreground-containing locations or use weighted loss.
