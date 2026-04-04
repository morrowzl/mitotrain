# Sprint 2 — Patch Sampler

**Status:** Ready for implementation. All decisions grounded in pre-sprint inspection findings.

**Goal:** Implement a patch sampler that yields balanced `(raw, label)` patch pairs drawn
from the full `mito_seg/s2` volume. The tool now prepares real training data.

**Exit condition:** `python train.py` samples real patches from real data, logs foreground
fraction per batch, and passes correctly shaped tensors to the stub model. At least 50% of
sampled patches contain foreground voxels.

---

## Decisions from Pre-Sprint Inspection

| Decision | Value | Source |
|----------|-------|--------|
| Label source | `labels/mito_seg/s2` | `inspect_mito_seg.py` — confirmed spatial alignment with EM |
| ⚠️ `mito_pred/s2` rejected | Labels extracellular space, not mitochondria | `inspect_roi.py` — red overlay confirmed below cell membrane |
| Binarization | `(seg > 0).astype(np.uint8)` | Instance IDs 47, 110, 138 → binary foreground |
| Class balance | 6.79% foreground | Workable; foreground-biased sampling sufficient |
| Imbalance strategy | Foreground-biased sampling only | Weighted loss deferred — orthogonal, add later if needed |
| Sampling scope | Full `mito_seg/s2` volume | Better diversity than confirmed ROI only |
| Volume shape | `(1592, 400, 3000)` | s2 confirmed; Y=400 is the tightest dimension |
| Patch size | `[NEEDS SHAPE TEST]` | Must verify funkelab U-Net valid output shape before finalizing |

---

## Tasks

### Step 0 — U-Net shape test (gate on patch size)

The funkelab U-Net uses valid convolutions — output is spatially smaller than input. Patch
size must be large enough that the output is not cropped to zero. This must be confirmed
before the sampler is written.

- [ ] Install funkelab U-Net:
  ```bash
  pip install git+https://github.com/funkelab/funlib.learn.torch.git
  ```
- [ ] Run shape test with candidate input sizes:
  ```python
  import torch
  from funlib.learn.torch.models import UNet

  model = UNet(
      in_channels=1,
      num_fmaps=12,
      fmap_inc_factor=5,
      downsample_factors=[(2,2,2),(2,2,2),(2,2,2)],
      kernel_size_down=[[[3,3,3],[3,3,3]]]*4,
      kernel_size_up=[[[3,3,3],[3,3,3]]]*3,
      num_fmaps_out=1,
      constant_upsample=True,
  )

  for size in [64, 96, 132, 148]:
      x = torch.zeros(1, 1, size, size, size)
      try:
          y = model(x)
          print(f"input {size}^3 → output {y.shape}")
      except Exception as e:
          print(f"input {size}^3 → FAILED: {e}")
  ```
- [ ] Record valid input→output shape pairs
- [ ] Select patch size that gives a non-trivial output (at least 4³)
- [ ] **Update `PATCH_SIZE` constant in `train.py` and sampler before proceeding**

> Note: 132³ is the documented safe starting size from Sprint 0 notes. If confirmed,
> use 132³ as `INPUT_PATCH_SIZE`. The U-Net output size will be smaller — record both.

### Step 1 — Update `data/loader.py`

Switch label layer from `mito_pred/s2` to `mito_seg/s2`.

```python
# The .n5 suffix is required — omitting it causes PathNotFoundError (Sprint 1 bug)
N5_PATH   = "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5"

# Change this constant:
SEG_LAYER = "labels/mito_seg/s2"   # was: labels/mito_pred/s2
```

Also update the ROI to the inspection-confirmed origin:
```python
ROI = (
    slice(480, 480 + PATCH_SIZE),
    slice(80,  80  + PATCH_SIZE),
    slice(2382, 2382 + PATCH_SIZE),
)
```

No other changes to loader logic needed — binarization `(seg > 0)` is unchanged.

### Step 2 — Implement `data/sampler.py`

Replace the Sprint 0 stub with a real foreground-biased patch sampler that draws from
the full `mito_seg/s2` volume.

**Design:**

The sampler operates in two phases per patch:
1. Sample a random origin within the valid volume bounds
2. Accept the patch if it contains at least `min_fg_frac` foreground voxels; reject and
   resample if not

To avoid loading the full volume into memory, the sampler should work with the zarr array
directly and load only the patch at the sampled origin.

**Function signature:**
```python
def sample_patches(
    em_array,            # zarr array: em/fibsem-uint16/s2
    seg_array,           # zarr array: labels/mito_seg/s2 (raw, not yet binarized)
    patch_size,          # int: edge length of cubic patch (e.g. 132)
    n_patches,           # int: number of patches to return
    min_fg_frac=0.05,    # float: minimum foreground fraction to accept patch
    max_attempts=50,     # int: max resamples before giving up on a patch
):
    """
    Sample n_patches paired (raw, label) crops from the full s2 volume.
    Returns list of (raw_patch, label_patch) tuples.
    raw_patch:   float32 numpy array, shape (1, patch_size, patch_size, patch_size)
    label_patch: uint8 numpy array,   shape (1, patch_size, patch_size, patch_size)
    """
```

**Implementation notes:**
- Volume shape is `(1592, 400, 3000)` — valid origin range per axis:
  - Z: `[0, 1592 - patch_size]`
  - Y: `[0, 400  - patch_size]`  ← tightest; will constrain large patch sizes
  - X: `[0, 3000 - patch_size]`
- **Derive bounds directly from array shape — do not calculate from nanometer coordinates.**
  Z, Y, and X have different physical voxel sizes (21 nm, 16 nm, 16 nm at s2). Any nm-based
  coordinate math must apply per-axis scale factors; a single divisor will produce wrong
  results. For the sampler, use `array.shape` directly and avoid nm arithmetic entirely.
- Use `np.random.randint` for origin sampling
- Binarize label patch with `(label_crop > 0)` after loading, before foreground check
- Apply percentile normalization to raw patch (same as loader: 1st/99th percentile)
- Log a warning if `max_attempts` is reached without finding a foreground patch — don't
  silently return an all-background patch
- Pass zarr arrays directly rather than pre-loaded numpy arrays — lazy loading is essential
  for the full volume

**Volume bounds check:** Y dimension is only 400 voxels. If `patch_size=132`, valid Y
origins are `[0, 268]`. If `patch_size=148`, valid Y origins are `[0, 252]`. Verify
the shape test result does not require a patch size that exhausts the Y dimension.

### Step 3 — Update `train.py`

Replace the stub sampler call with the real sampler. Pass zarr array handles rather than
pre-loaded numpy arrays.

```
[1/7] Loading subvolume...        shape: (1592, 400, 3000)  ← full volume handle
[2/7] Generating binary mask...   deferred to sampler        ← note change
[3/7] Sampling patches...         n=8, fg_frac: X.XX        ← real foreground fraction
[4/7] Instantiating model...      params: 0 (stub)          ← still stub
[5/7] Forward pass...             output shape: (...)       ← updated for real patch size
[6/7] Computing loss...           loss: X.XXXX (stub)       ← still stub
[7/7] Saving visualization...     outputs/slice_preview.png ← update to show sampled patch
```

Step 2 log should print the mean foreground fraction across all sampled patches.

### Step 4 — Update `utils/visualize.py`

Update to show a sampled patch rather than the fixed ROI subvolume:
- Pick the patch with the highest foreground fraction from the batch
- Show middle Z slice of that patch: raw EM left, binary mask right
- Add foreground fraction to the title

### Step 5 — Sanity checks

- [ ] At least 50% of sampled patches contain foreground (min_fg_frac=0.05 enforcement working)
- [ ] No patch has foreground fraction = 1.0 (would indicate label flood-fill)
- [ ] Raw patches are float32 in [0, 1]; label patches are uint8 in {0, 1}
- [ ] **Visual check: red overlay in slice_preview.png falls on the bright-outlined oval
  structures in the EM — not on the uniform gray region below the cell membrane.**
  Statistical foreground counts alone are insufficient — Sprint 1 showed that a nonzero
  voxel count can still represent the wrong structure entirely.
- [ ] Y-axis origin never exceeds `400 - patch_size` (bounds check working)

---

## Commit Plan

| Commit | Message |
|--------|---------|
| 1 | `sprint2: run U-Net shape test, record valid patch sizes` |
| 2 | `sprint2: switch loader to mito_seg/s2, update ROI` |
| 3 | `sprint2: implement foreground-biased patch sampler` |
| 4 | `sprint2: wire sampler into train.py, update visualization` |
| 5 | `sprint2: sanity checks pass, slice_preview updated` |

---

## Notes and Open Questions Carried Forward

- **Weighted loss:** `nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w]))` where `w ≈ 1/fg_frac`.
  Fully orthogonal to sampling strategy. Add in Sprint 4 if training loss plateaus.

- **Full-volume diversity:** The sampler draws from all of `(1592, 400, 3000)` but `mito_seg`
  coverage may not be uniform across the volume. If most of the volume has zero foreground,
  the sampler will spend many attempts rejecting background patches. Monitor rejection rate
  in Sprint 3 and consider pre-computing a foreground location index if it becomes a
  bottleneck.

- **mito_pred investigation (backlog):** `mito_pred/s2` labeled extracellular space rather
  than mitochondria at the Sprint 1 ROI. The relationship between `mito_pred` and `mito_seg`
  across the full volume is not yet understood. Not blocking Sprint 2 but worth revisiting.

- **GT crops (backlog):** Label encoding resolved (IDs 3+4+5 = mito). Coordinate alignment
  to s2 EM still needed. Deferred to evaluation sprint.

- **Patch size dependency:** `PATCH_SIZE` is a constant that flows through loader, sampler,
  model, and train.py. Once confirmed by shape test, define it in a single location
  (top of `train.py` or a `config.py`) and import everywhere — do not hardcode separately
  in each module.
