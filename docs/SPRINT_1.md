# Sprint 1 — Real Data In, Real Labels Out

**Status:** Ready for implementation. All `[NEEDS VERIFICATION]` items resolved.

**Goal:** Load a real subvolume of raw EM and a corresponding binary mitochondria mask from S3.
Visualize a 2D slice of each side by side. The tool now does one genuinely useful thing: it
shows you the data.

**Exit condition:** `python train.py` produces a saved side-by-side slice visualization of
real raw EM and a real binary mito mask, with correct shapes printed to console.

---

## Decisions from SPRINT_1_RESEARCH.md

| Decision | Value | Rationale |
|----------|-------|-----------|
| Bucket | `janelia-cosem-datasets` | Both EM and mito_pred in one place; no coordinate alignment needed |
| Dataset | `jrc_hela-2` | Recommended starting point by project lead and community |
| Raw EM path | `em/fibsem-uint16/s2` | 16 nm/vox — practical for CPU, mito still clearly resolvable |
| Raw EM shape | `(1592, 400, 3000)` | Confirmed via explore.py |
| Raw EM dtype | `uint16` | Confirmed; actual signal range ~10000–20000, not 0–65535 |
| Label path | `labels/mito_pred/s2` | Confirmed exists; shape `(1592, 400, 3000)`, dtype `uint8`, chunks `(96, 96, 96)` |
| Label values | `{0, 1}` | Binary — verify with `np.unique()` on first crop; no thresholding needed if confirmed |
| Label source | `mito_pred` (model prediction) | Simpler than GT crops; GT crops deferred to evaluation sprint |
| ROI size | `96³` | Matches `mito_pred/s2` chunk size — avoids partial chunk fetches |
| ROI origin (s2) | `(z=1800, y=90, x=2390)` | Derived from neuroglancer coordinates (nm ÷ 16); confirmed mitochondria-dense region |
| Normalization | Percentile-based | Raw EM signal clustered in ~10000–20000 range; divide by 65535 would compress contrast |
| zarr pin | `zarr<3` | N5FSStore deprecated in v3; pin to avoid silent breakage |

---

## Tasks

### Step 0 — Pre-implementation checks

All items below were verified before implementation. Documented here for reference.

- [x] **VERIFIED** `labels/mito_pred/s2` exists in jrc_hela-2 — shape `(1592, 400, 3000)`,
  dtype `uint8`, chunks `(96, 96, 96)`
- [x] **VERIFIED** `mito_pred/s2` shape matches `em/fibsem-uint16/s2` exactly — no coordinate
  scaling needed between raw EM and labels
- [ ] **CONFIRM AT RUNTIME** Unique values in `mito_pred/s2` crop are `{0, 1}` not `{0, 255}`:
  ```python
  print(np.unique(label_crop))  # expected: [0 1]
  ```
  If `{0, 255}`, threshold at 127 in loader before returning.
- [x] **VERIFIED** explore.py bugs fixed (groundtruth path, mito_pred path, PYTHONUTF8)
- [ ] **CONFIRM AT RUNTIME** Raw EM crop min/max and 1st/99th percentile — print before
  normalizing to confirm signal is in expected range (~10000–20000):
  ```python
  print(raw_crop.min(), raw_crop.max(), np.percentile(raw_crop, [1, 99]))
  ```

### Step 1 — Implement `data/loader.py`

Replace the Sprint 0 stub with a real implementation.

**Function signature (keep unchanged from stub):**
```python
def load_subvolume(dataset, layer, roi):
    """
    Load a subvolume from the janelia-cosem-datasets S3 bucket.

    Args:
        dataset: str, e.g. 'jrc_hela-2'
        layer:   str, e.g. 'em/fibsem-uint16/s2' or 'labels/mito_pred/s2'
        roi:     tuple of slices, e.g. (slice(0,64), slice(0,64), slice(0,64))

    Returns:
        numpy array of the requested subvolume
    """
```

**Implementation notes:**
- Use `zarr.N5FSStore` with `anon=True` — no credentials needed
- Use `dask.array.from_array(zdata, chunks=zdata.chunks)` for lazy loading
- Call `.compute()` only on the ROI slice, not the full array
- Raw EM is `uint16` — normalize to float32 in [0, 1] using **percentile-based normalization**,
  not dtype max. The actual signal range is ~10000–20000, not 0–65535. Dividing by 65535
  compresses all contrast into a narrow band:
  ```python
  raw = raw.astype(np.float32)
  p_low, p_high = np.percentile(raw, [1, 99])
  raw = np.clip((raw - p_low) / (p_high - p_low), 0, 1)
  ```
- `mito_pred` should be returned as `uint8` or `bool` — do not normalize
- `mito_pred/s2` chunks are `(96, 96, 96)` — use a ROI that aligns to this to avoid
  fetching extra chunks unnecessarily
- Pin `zarr<3` in requirements.txt

**Hardcoded constants for Sprint 1** (will move to config in Sprint 4):
```python
BUCKET    = "janelia-cosem-datasets"
DATASET   = "jrc_hela-2"
EM_LAYER  = "em/fibsem-uint16/s2"
SEG_LAYER = "labels/mito_pred/s2"

# ROI origin derived from neuroglancer coordinates (nm ÷ 16 for s2)
# Confirmed mitochondria-dense region from visual inspection
ROI = (
    slice(1800, 1896),  # Z: 96 voxels
    slice(90,   186),   # Y: 96 voxels  (note: Y max is 400 at s2 — stay well within bounds)
    slice(2390, 2486),  # X: 96 voxels
)
```

### Step 2 — Update `train.py` steps 1 and 2

Replace stub calls with real loader calls. Keep all other steps (3–7) as stubs unchanged.

```
[1/7] Loading subvolume...        shape: (1, 96, 96, 96)   ← real data
[2/7] Generating binary mask...   mito voxels: N            ← real count
[3/7] Sampling patches...         n=8, shape: (1, 32, 32, 32)   ← still stub
...
```

Step 2 (binary mask) should now report a real mito voxel count. If count is 0, the ROI
origin may have drifted from the mitochondria-dense region — double-check the slice
coordinates against the neuroglancer-verified origin above.

### Step 3 — Update `utils/visualize.py`

Replace the blank figure stub with a real two-panel matplotlib visualization:
- Left panel: 2D slice of raw EM (middle Z slice of the subvolume)
- Right panel: 2D slice of binary mito mask (same Z)
- Add a colormap for the mask (e.g. red overlay or binary cmap)
- Save to `outputs/slice_preview.png`

### Step 4 — Sanity checks

Before committing, verify:
- [ ] Raw EM slice looks like grayscale electron microscopy (membrane contrast visible)
- [ ] Mito mask slice contains non-zero regions that plausibly overlap with dense structures in EM
- [ ] Console output shows realistic mito voxel count (not 0, not 100%)
- [ ] `outputs/slice_preview.png` is committed to the repo as a visual record

---

## Commit Plan

| Commit | Message |
|--------|---------|
| 1 | `sprint1: verify mito_pred/s2 exists, update explore.py` |
| 2 | `sprint1: implement data/loader.py with real S3 access` |
| 3 | `sprint1: update train.py steps 1-2 with real data` |
| 4 | `sprint1: implement real slice visualization` |
| 5 | `sprint1: sanity checks pass, slice_preview.png committed` |

---

## Notes and Open Questions Carried Forward

- **GT label encoding** (open question 1 from research): the `labels/all` arrays in the
  publications bucket use `uint64` instance IDs. It is not yet known which IDs map to
  mitochondria. This must be resolved before GT crops can be used for evaluation in Sprint 5+.
  Leave as a backlog item.

- **Coordinate alignment** (open question 2): GT crops are at s0 resolution; aligning to s2
  EM requires world-coordinate offset mapping. Deferred — not needed until evaluation sprint.

- **zarr v3 migration**: `N5FSStore` is deprecated in zarr v3. Pin `zarr<3` now. Track
  whether `zarr-python` v3 provides a clean N5 replacement before this becomes a blocker.

- **Patch size**: ROI updated to 96³ to align with `mito_pred/s2` chunk size. Will be
  revised again in Sprint 2 once the funkelab U-Net input shape constraint (132³ minimum)
  is confirmed via shape test.
