# Lessons Learned

Running log of findings that caused adjustments from preliminary plans.
Updated after each sprint.

---

## Sprint 0 / Pre-Sprint 1

### 1. Verify S3 paths before planning against them
The research doc assumed `volumes/labels/0003` was the correct path for ground truth crops
in the publications bucket. The actual path is `volumes/groundtruth/0003`. Similarly,
`labels/mito/s0` does not exist — the correct names use underscores (`mito_pred`, `mito_seg`).
Both were caught by `explore.py` before any code was written, but they illustrate why
stub-first planning docs with explicit path verification steps are worth the overhead.

**Rule:** Never write a loader or sampler against a path that has not been confirmed to exist
by running code against the actual store.

---

### 2. The `.n5` suffix in the S3 path is required
The S3 path for jrc_hela-2 must include the `.n5` container suffix:
```
s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5
```
Omitting it raises a `PathNotFoundError`. This is not obvious from the bucket URL alone.

**Rule:** Always include the `.n5` suffix and add a comment in loader constants explaining why.

---

### 3. Voxel sizes are not uniform across axes (anisotropy)
We assumed a single conversion factor (16 nm/vox) when converting neuroglancer nanometer
coordinates to s2 voxel coordinates. The Z axis is physically 5.24 nm at s0 (~21 nm at s2),
while X and Y are 4.0 nm (~16 nm at s2). Applying a uniform divisor caused the Sprint 1
ROI to overflow the Z dimension entirely (z=1800 vs max z=1592).

The per-axis voxel sizes are available in the array attributes:
```
transform.scale: [5.24, 4.0, 4.0]  (Z, Y, X) at s0
```

**Rule:** Any coordinate conversion from world space (nm) to voxel space must use per-axis
scale factors. For sampler bounds, use `array.shape` directly rather than nm arithmetic.

---

## Sprint 1

### 4. `mito_pred/s2` does not label mitochondria at this ROI
The `mito_pred/s2` layer was described in the README as "a scalar field of uint8 values
HIGH inside an object class and LOW outside" — implying soft predictions (0–255). At s2,
the actual values were {0–7}, and visual inspection confirmed the overlay labeled the
extracellular space below the cell membrane, not the mitochondria visible in the EM.

`mito_pred` and `mito_seg` had zero spatial overlap at the Sprint 1 ROI. They are not
interchangeable. `mito_seg/s2` correctly overlaid the visible mitochondria and was adopted
as the label source for Sprint 2.

**Rule:** Do not assume downsampled derived arrays behave semantically the same as their
s0 counterparts. Always visually verify label alignment with the EM before using a layer
as a training signal.

---

### 5. Visual inspection is irreplaceable — statistics alone are not sufficient
The Sprint 1 slice preview showed a nonzero foreground voxel count (508 voxels), which
passed the statistical sanity check. However, visual inspection of the overlay revealed
the red region was entirely below the cell membrane — outside the cell — not on the
mitochondria visible in the EM. A count of 508 foreground voxels was technically correct
and completely misleading.

**Rule:** Every sprint exit condition must include a visual check that the label overlay
lands on the correct structures in the EM — not just that foreground voxels are present.

---

### 6. Class imbalance figures depend entirely on which label source is used
`mito_pred/s2` showed 0.06% foreground at the Sprint 1 ROI. After switching to `mito_seg/s2`
at the same location, foreground was 6.79% — a 100× difference. The two layers label
different things. The 0.06% figure was not a property of mitochondria density in this
region; it was a property of the wrong label source.

**Rule:** Never reason about class balance, foreground density, or patch sampling strategy
until the label source has been visually confirmed to label the intended structure.

---

### 7. Downsampled label arrays may have different value semantics than s0
`mito_pred/s0` was described as binary {0,1}. At s2, values {0–7} were observed — small
integers that coincide with the GT class ID table from the README (1=ECS, 2=PM, 3=mito
membrane, etc.). The downsampling process appears to corrupt the soft-prediction
interpretation of `mito_pred`, producing what look like class ID mixtures instead.

**Rule:** Always run `np.unique()` on a crop of any label array before writing binarization
logic. Do not assume the value encoding at s2 matches the documentation for s0.

---

## Sprint 4.1

### 8. Foreground check must target the center-crop region, not the full input patch
The sampler demanded 5% foreground in the 132³ input patch, but BCE loss is computed only
on the center 40³ output region. A 132³ patch with 5% foreground has ~0.14% foreground in
a random 40³ subregion — effectively all-background labels for the model.

**Rule:** When input patch size ≠ output region size, always evaluate foreground fraction
on the center crop that matches the loss region. Add `OUTPUT_SIZE` as an importable constant
alongside `PATCH_SIZE` so the sampler and training code stay in sync.

---

### 9. Write summaries from actual console output, not expected output
Sprint 4's summary document showed a fabricated decreasing loss curve (0.6931 → 0.6714 → 0.6421).
The actual console output was flat 0.6931 throughout. The summary was generated from
"expected" behavior rather than observed output.

**Rule:** Sprint summary documents must be written from copy-pasted console output.
Never substitute expected results for actual results.
