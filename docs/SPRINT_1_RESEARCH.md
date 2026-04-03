# Sprint 1 Research — explore.py Output and Analysis

**Run date:** 2026-04-02  
**Command:** `PYTHONUTF8=1 python explore.py`  
**Exit code:** 0

> Note: A first run failed with `UnicodeEncodeError` (cp1252 cannot encode zarr tree box-drawing
> characters). Fixed by setting `PYTHONUTF8=1`. Also note zarr v2 emits a `FutureWarning` about
> `N5FSStore` being deprecated in v3 — harmless for now, but see "Open questions" below.

---

## Key Answers

### Which bucket to use?

Both buckets are accessible anonymously.

| Bucket | Purpose | Use for |
|--------|---------|---------|
| `janelia-cosem-publications` | Ground truth crops (hand-annotated, multi-organelle) | Training labels |
| `janelia-cosem-datasets` | Full-volume EM + automated segmentations/predictions | Raw EM input |

**Decision for Sprint 1:** Load raw EM from `janelia-cosem-datasets` (`em/fibsem-uint16`).
Load training labels from the publications bucket ground truth crops
(`volumes/groundtruth/0003/<crop>/labels/all`).

### What are the ground truth crop names and shapes?

Path: `janelia-cosem-publications/.../volumes/groundtruth/0003/`  
All arrays are `uint64` instance segmentation labels.

> **Bug in explore.py:** The script checked `volumes/labels/0003` — the correct path is
> `volumes/groundtruth/0003`. Fix this before Sprint 1.

| Crop | Shape (Z, Y, X) | Notes |
|------|----------------|-------|
| crop1 | (200, 1000, 1000) | |
| crop3 | (500, 800, 800) | |
| crop4 | (476, 600, 600) | also has centrosome sublabel + masks |
| crop6 | (500, 500, 500) | |
| crop7 | (160, 600, 600) | |
| crop8 | (200, 400, 400) | |
| crop9 | (106, 200, 200) | smallest crop |
| crop13 | (220, 320, 320) | |
| crop14 | (130, 300, 300) | |
| crop15 | (128, 300, 300) | |
| crop16 | (400, 400, 400) | also has nucleolus + ribosomes sublabels |
| crop18 | (220, 400, 400) | |
| crop19 | (110, 300, 300) | |
| crop23 | (500, 500, 500) | |
| crop28 | (400, 400, 400) | |
| crop54 | (400, 400, 400) | |
| crop55 | (400, 400, 400) | |
| crop56 | (400, 400, 400) | |
| crop57 | (400, 400, 400) | |
| crop58 | (400, 400, 400) | |
| crop59 | (400, 400, 400) | |
| crop94 | (400, 400, 400) | |
| crop95 | (400, 400, 400) | |
| crop96 | (400, 400, 400) | |
| crop113 | (500, 1000, 1000) | also has ribosomes sublabel |

### GT label encoding — RESOLVED

**Source:** `heinrich-2021a` README from `s3://janelia-cosem-publications/heinrich-2021a/`

The `labels/all` array in each crop uses a fixed integer ID scheme consistent across all
crops and datasets. IDs are **semantic class labels**, not arbitrary instance IDs.

Mitochondria are represented by the following IDs:

| ID | Class | Combined label |
|----|-------|----------------|
| 3 | Mito membrane | Mito = IDs 3 + 4 + 5 |
| 4 | Mito lumen | |
| 5 | Mito ribosome | |

To extract a binary mito mask from a GT crop:
```python
mito_mask = np.isin(labels_all, [3, 4, 5]).astype(np.uint8)
```

Selected other IDs for reference (full table in README):

| ID | Class |
|----|-------|
| 1 | Extracellular space (ECS) |
| 2 | Plasma membrane |
| 6 | Golgi membrane |
| 7 | Golgi lumen |
| 8 | Vesicle membrane |
| 9 | Vesicle lumen |
| 16–23 | ER and nuclear envelope variants |
| 35 | Cytosol |

This resolves open question 1. GT crops can now be used for evaluation once coordinate
alignment (open question 2) is addressed.

### What downsampled resolution level is practical for CPU?

Raw EM (`em/fibsem-uint16`):

| Level | Shape | Voxel size | Voxels |
|-------|-------|-----------|--------|
| s0 | (6368, 1600, 12000) | 4 × 4 × 5.24 nm | 122B |
| s1 | (3184, 800, 6000) | 8 × 8 × 10.5 nm | 15B |
| s2 | (1592, 400, 3000) | 16 × 16 × 21 nm | 1.9B |
| s3 | (796, 200, 1500) | 32 × 32 × 42 nm | 239M |
| s4 | (398, 100, 750) | 64 × 64 × 84 nm | 30M |

> **Note on voxel size anisotropy:** Z voxel size at s0 is 5.24 nm vs 4.0 nm for X/Y. At s2
> this becomes ~21 nm in Z vs ~16 nm in X/Y. Coordinate conversions from world space (nm)
> to voxel space must use per-axis scale factors, not a single divisor. This caused an
> out-of-bounds ROI in Sprint 1 (z=1800 exceeded max z=1592).

**Recommendation:** Start with **s2** (16 nm/vox X/Y, 21 nm/vox Z). It's small enough for
lazy remote access without full download, mitochondria are still clearly resolvable at 16 nm,
and patch sampling at 64–132 voxels covers a meaningful volume. s4 is very coarse (64 nm)
and may lose fine mito structure; s0/s1 are impractical on CPU without significant cropping.

The ground truth crops in the publications bucket appear to be at **s0 resolution** (4–5 nm/vox)
based on their voxel counts. Aligning GT crops to s2 raw EM will require coordinate scaling.

### Mito label paths (datasets bucket)

> **Bug in explore.py:** The script queried `labels/mito/s0` which does not exist as an array —
> it's a nested group. The actual label names use underscores:

| Path | Shape | dtype | Description |
|------|-------|-------|-------------|
| `labels/mito_seg/s0` | (6368, 1600, 12000) | uint16 | Instance segmentation |
| `labels/mito_seg/s4` | (398, 100, 750) | uint16 | 16× downsampled |
| `labels/mito_pred/s0` | (6368, 1600, 12000) | uint8 | Soft prediction (0–255) per README |
| `labels/mito_pred/s2` | (1592, 400, 3000) | uint8 | 4× downsampled, chunks (96,96,96) — **Sprint 1 target** |
| `labels/mito_pred/s4` | (398, 100, 750) | uint8 | 16× downsampled |

> **Important — mito_pred value encoding:** Per the `heinrich-2021a` README, prediction volumes
> are described as "a scalar field of `uint8` values that are **high** at a location inside an
> instance of an object class and **low** at a location outside." This suggests `mito_pred/s0`
> is a soft prediction (0–255), not a strict binary {0,1} array. The Sprint 1 summary observed
> values {0,1,2,...,9} in `mito_pred/s2`, which may reflect either:
> (a) downsampling artifacts mixing soft prediction values from adjacent voxels, or
> (b) the array being a multi-class label rather than a single-class prediction
>
> This is **unresolved** and is the primary open question for Sprint 2 research.
> See `inspect_binarization.py`.

### Does fibsem_tools work?

Not tested — `fibsem_tools` is not installed. Given that raw zarr access works cleanly
for both buckets, `fibsem_tools` is likely unnecessary for Sprint 1.

---

## Open Questions

1. ~~**GT label encoding**~~ — **RESOLVED.** Mito = IDs 3, 4, 5 in `labels/all`. Source:
   `heinrich-2021a` README from `s3://janelia-cosem-publications/heinrich-2021a/`.

2. **Coordinate alignment:** The GT crops are at s0 resolution. To pair GT labels with
   downsampled EM (s2), need to determine the crop offsets in world coordinates and map
   to s2 voxel coordinates. Deferred to evaluation sprint.

3. **mito_pred vs GT crops:** Decision: use `mito_pred/s2` for Sprint 1 (simpler), switch
   to GT crops for evaluation.

4. ~~**explore.py bugs**~~ — **RESOLVED.** Fixed before Sprint 1.

5. **mito_pred binarization:** Are `mito_pred` values at s0 truly binary {0,1}, or a soft
   prediction (0–255)? If soft, what threshold is appropriate? Do intermediate values at s2
   (observed: {0–9}) reflect downsampling artifacts or multi-class data?
   → To be answered by `inspect_binarization.py` before Sprint 2 planning is finalized.

6. **ROI alignment:** Sprint 1 visualization showed red overlay at the cell boundary rather
   than over the visually apparent mitochondria. Is this a ROI placement issue, a label
   quality issue at 16 nm, or a genuine spatial mismatch between EM and mito_pred at s2?
   → To be answered by `inspect_roi.py` before Sprint 2 planning is finalized.

---

## Raw EM Attributes (s0, datasets bucket)

```
path:   em/fibsem-uint16/s0
shape:  (6368, 1600, 12000)
dtype:  uint16
chunks: (64, 64, 64)
attrs:
  pixelResolution: {dimensions: [4.0, 4.0, 5.24], unit: "nm"}
  transform:
    axes:      [z, y, x]
    scale:     [5.24, 4.0, 4.0]
    translate: [0.0, 0.0, 0.0]
    units:     [nm, nm, nm]
```

> **Note on axis ordering:** The `transform.scale` field lists axes as [z, y, x] = [5.24, 4.0, 4.0].
> Z is the slowest-changing axis and has a larger physical voxel size. Any coordinate math
> must apply the correct scale per axis.
