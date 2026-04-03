# `mito_seg` Inspection Findings

**Date:** 2026-04-03  
**Script:** `inspect_mito_seg.py`  
**Purpose:** Characterize `mito_seg/s2` as a candidate ground-truth source for the Sprint 2 patch sampler, using the ROI origin confirmed in pre-sprint inspection.

---

## ROI

```python
ROI_Z = slice(480, 576)   # 96 voxels
ROI_Y = slice(80,  176)   # 96 voxels
ROI_X = slice(2382, 2478) # 96 voxels
```

---

## Value Encoding

| Property | Value |
|----------|-------|
| dtype | uint16 |
| Unique values | `{0, 47, 110, 138}` |
| Foreground voxels (`> 0`) | 60,063 / 884,736 = **6.79%** |
| Best Z slice | z=560, 2,494 mito voxels |

Values are instance IDs — three distinct mitochondrial objects are present in the ROI. Binarization via `seg > 0` collapses all instances to a single foreground class, which is correct for binary segmentation training.

---

## Comparison: `mito_seg` vs `mito_pred`

| Layer | Foreground % | Unique values | Interpretation |
|-------|-------------|---------------|----------------|
| `mito_seg/s2` | **6.79%** | `{0, 47, 110, 138}` | Instance segmentation — hand-curated or high-quality model |
| `mito_pred/s2` | 0.26% | `{0–7}` | Sparse prediction — low recall at this ROI |

`mito_seg` covers ~26× more foreground voxels at the same location. The zero overlap found between the two layers in `inspect_binarization.py` is explained by `mito_pred` labeling almost nothing here while `mito_seg` labels the actual mitochondria visible in the EM.

**Conclusion: use `mito_seg/s2` as ground truth for Sprint 2.**

---

## Available Scales

`labels/mito_seg` exposes five resolution levels: `s0, s1, s2, s3, s4`. Sprint 2 will train at s2 (16 nm effective resolution). Higher-resolution training at s0 or s1 is possible in later sprints if compute allows.

---

## Visual Output

`outputs/inspect_mito_seg.png` — three panels at the best Z slice (z=560):

| Panel | Contents |
|-------|----------|
| Left | Raw EM only |
| Center | EM + red mito_seg overlay (α=0.5) |
| Right | Binary seg mask only |

---

## Sprint 2 Implications

- **Label source:** Switch loader from `labels/mito_pred/s2` to `labels/mito_seg/s2`
- **Binarization:** `(seg > 0).astype(np.uint8)` — same logic, different layer
- **Class balance:** 6.79% foreground is workable; weighted loss or foreground-biased patch sampling still advisable but less critical than at 0.06%
- **ROI confirmed:** `Z: slice(480, 576)` captures the dense mito band (best slice z=560 falls within it)
