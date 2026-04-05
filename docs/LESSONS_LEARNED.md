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

---

## Sprint 4.2

### 10. The funkelab U-Net default activation (ReLU) is incompatible with BCEWithLogitsLoss
The funkelab U-Net defaults to `activation='ReLU'` on all layers including the final output
layer. `BCEWithLogitsLoss` expects raw logits — values that can be positive or negative.
ReLU clamps all outputs to `[0, ∞)`, preventing negative logits and causing the model to
output near-zero values at initialization that never move. Loss was exactly ln(2) = 0.6931
for all batches across multiple epochs.

The fix is `activation=None` in the `UNet(...)` constructor. This is not documented
prominently in the funkelab README and must be set explicitly.

**Rule:** When using a U-Net (or any model) with a custom loss function, verify the final
layer activation is compatible. For `BCEWithLogitsLoss`, the model must output raw logits
with no final activation. Print `model` to inspect the full layer tree before training.

---

### 11. Gradient explosion without BatchNorm requires explicit clipping
With no BatchNorm layers and `lr=1e-3`, gradients exploded within the first epoch (outputs
reaching ±5000, loss ~1234). The 93M parameter model has no internal normalization to
constrain gradient magnitudes.

Fix: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)` after
`loss.backward()`, combined with reducing `lr` to `1e-4`.

**Rule:** For deep networks without BatchNorm, always add gradient clipping. Start with
`max_norm=1.0` and increase if loss fails to move; `max_norm=10.0` worked here.

---

### 12. Class imbalance causes background collapse without weighted loss
Even with foreground-biased sampling, the model learned to output uniformly low probabilities
(background-leaning) for all voxels. Adding `pos_weight=4.0` to `BCEWithLogitsLoss`
penalizes missed foreground predictions more heavily and prevented collapse.

**Rule:** For binary segmentation with imbalanced classes, always use `pos_weight` in
`BCEWithLogitsLoss`. A good starting value is `(1 - fg_frac) / fg_frac` — approximately
the ratio of background to foreground voxels.

---

### 13. Anomalous backward pass timing in later epochs (cause unknown)
Epoch 25 backward pass took 940s vs typical 120–175s. Did not recur in epochs 26–30.
Epoch 19 sampling took 2073s — MEM dropped from 4.64 GB to 2.88 GB, suggesting OS memory
reclamation. Neither was reproduced or root-caused.

**Rule:** Log per-phase timing every epoch. Anomalies that don't recur are likely OS/network
events; anomalies that grow monotonically indicate a code issue (e.g. gradient graph
accumulation from missing `zero_grad`).

---

## Sprint 5

### 14. Verify holdout ROI contains foreground before committing it as a constant
The originally committed `HOLDOUT_ROI` at Z=800–932, X=1200–1332 had zero foreground
voxels in `mito_seg/s2`. The Sprint 4.1 plan included a verification step for this but
it was not executed before the constant was committed. Inference produced a blank ground
truth panel and a meaningless comparison.

**Rule:** Any ROI used for evaluation must be verified to contain foreground before being
committed as a constant. Run `(seg[roi] > 0).sum()` and confirm non-zero before proceeding.

---

### 15. Model output dtype must match model weight dtype (float32 vs float64)
`predict.py` failed with `RuntimeError: Input type (double) and bias type (float) should
be the same`. `np.percentile` returns `float64`, which propagated through normalization
into the input tensor despite an earlier `.astype(np.float32)` call.

Fix: `.float()` on the tensor after `torch.from_numpy(...)`:
```python
x = torch.from_numpy(raw_norm[np.newaxis, np.newaxis]).float()
```

**Rule:** Always call `.float()` explicitly when constructing input tensors for PyTorch
models trained in float32. Do not rely on numpy dtype propagation.

---

### 16. A narrow probability range means the model has not learned spatial discrimination
The trained model produced probabilities in the range 0.2657–0.3240 — a span of only 0.058
across the entire 40³ output volume. Any threshold in this range produces either near-zero
or near-total foreground predictions, neither of which is meaningful.

This is not an inference bug. It reflects insufficient training — 240 gradient steps on a
93M parameter model without BatchNorm is not enough to learn to discriminate foreground
from background spatially. The probability map (contrast-stretched) shows learned spatial
structure, confirming the pipeline is functional.

**Rule:** When model output probabilities span a narrow range, do not select an arbitrary
threshold for the portfolio visualization. Show the contrast-stretched probability map
instead — it honestly represents what the model learned at this stage of training.
