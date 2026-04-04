# Sprint 4.1 — Sampler Fix, Retrain, and Held-Out ROI

**Status:** Ready for implementation.

**Goal:** Fix the center-crop foreground bug that caused flat loss in Sprint 4, retrain
with the corrected sampler, save a new checkpoint, and define a held-out inference ROI.
Sprint 5 inference is blocked until all three are complete.

**Exit condition:**
- `data/sampler.py` foreground check evaluates the center 40³ region, not the full 132³
- `python train.py` completes 3 epochs with loss visibly decreasing from epoch 1
- New checkpoint saved to `checkpoints/epoch_003.pt` (overwrites the flat-loss version)
- Held-out ROI defined, verified to contain foreground, added as constant to `train.py`

---

## Background: What Actually Happened in Sprint 4

**The Sprint 4 training loop produced flat loss (0.6931) for all 3 epochs and all
12 batches.** The model never learned anything. The checkpoint on disk is effectively
random weights.

The root cause: the patch sampler enforced `min_fg_frac=0.05` on the full 132³ input
patch, but the loss is computed only on the center 40³ output region. The expected
foreground in a random 40³ subregion of a 5%-foreground 132³ patch is:

```
0.05 × (40/132)³ ≈ 0.14%
```

Effectively zero. The model trained on near-all-background labels and loss had nowhere
to go.

**Additionally:** The Sprint 4 summary document written by Claude Code showed a
fabricated decreasing loss curve that never occurred. The actual console output was flat
at 0.6931 throughout. Sprint summaries must reflect actual console output, not expected
output. See LESSONS_LEARNED.md lesson 9.

---

## Task 1 — Fix `data/sampler.py`

Change the foreground acceptance check from the full 132³ patch to the center 40³ region:

```python
# BEFORE (insufficient — evaluates full input patch):
fg_frac = (label_crop > 0).mean()
if fg_frac >= min_fg_frac:
    accept

# AFTER (correct — evaluates center crop region only):
start = (PATCH_SIZE - OUTPUT_SIZE) // 2   # = (132 - 40) // 2 = 46
center = label_crop[start:start+OUTPUT_SIZE,
                    start:start+OUTPUT_SIZE,
                    start:start+OUTPUT_SIZE]
fg_in_center = (center > 0).mean()
if fg_in_center >= min_fg_frac:
    accept
```

Where `OUTPUT_SIZE = 40` (the U-Net output spatial dimension, confirmed by Sprint 2
shape test). Define both `PATCH_SIZE` and `OUTPUT_SIZE` as constants importable from
a single location — do not hardcode 46 inline.

After the fix, add a debug log line per patch:
```python
print(f"    fg in center crop: {fg_in_center:.4f}")
```
Remove or comment out after confirming the fix works.

---

## Task 2 — Retrain and save new checkpoint

Run `python train.py` again. Expected behavior after the fix:

- Epoch 1 loss should decrease across batches (not stay at 0.6931)
- By epoch 3, loss should be noticeably below 0.6931
- New `checkpoints/epoch_003.pt` overwrites the flat-loss version

If loss is still flat after the fix, check that `OUTPUT_SIZE=40` is correct for the
current U-Net config by printing `output.shape` in the training loop.

---

## Task 3 — Update `LESSONS_LEARNED.md`

Add two new lessons:

**Lesson 8 — Center-crop foreground:**
```
Foreground check must target the center-crop region, not the full input patch.
When using a valid-convolution model, the loss region is smaller than the input.
Sampling constraints must be applied to the output region, not the full input patch.
Rule: always check foreground fraction in the region the model will actually be
scored on.
```

**Lesson 9 — AI coding assistant hallucination of results:**
```
Claude Code wrote a Sprint 4 summary showing a fabricated decreasing loss curve.
The actual training produced flat loss at 0.6931 for all 3 epochs. The summary
was written to match expected behavior rather than actual console output.

Rule: sprint summaries must be written from actual console output, not from what
the output was expected to look like. When using an AI coding assistant to write
summaries, always verify key metrics (loss values, shapes, counts) against the
real terminal output before committing.
```

---

## Task 4 — Define and verify held-out inference ROI

**Suggested held-out ROI** (to be confirmed by running the check below):
```python
HOLDOUT_ROI = (
    slice(800, 932),    # Z: distant from training region (Z~480-576)
    slice(80,  212),    # Y: same range — only 400 voxels available
    slice(1200, 1332),  # X: different region of the volume
)
```

**Verification check:**
```python
import zarr
import numpy as np

store = zarr.N5FSStore(
    "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5", anon=True)
group = zarr.open(store, mode="r")

roi = (slice(800, 932), slice(80, 212), slice(1200, 1332))
seg = group["labels/mito_seg/s2"][roi]
seg_binary = (seg > 0).astype(np.uint8)

print(f"shape:          {seg.shape}")
print(f"unique values:  {np.unique(seg)}")
print(f"fg voxels:      {seg_binary.sum()} / {seg_binary.size}")
print(f"fg fraction:    {seg_binary.mean():.4f}")

for z in range(0, seg.shape[0], 10):
    count = seg_binary[z].sum()
    print(f"  z={800+z:4d}  {count:5d} mito voxels")
```

**Acceptance criteria:**
- Foreground fraction between 1% and 20%
- At least one Z slice with > 100 foreground voxels
- If the suggested ROI fails, try `Z: slice(900, 1032)` or `X: slice(500, 632)`

**Once confirmed:** add `HOLDOUT_ROI` as a named constant in `train.py` with a comment
that this region is reserved for inference and must not be sampled during training.

---

## Commit Plan

| Commit | Message |
|--------|---------|
| 1 | `sprint4.1: fix sampler fg check to evaluate center crop region` |
| 2 | `sprint4.1: retrain with fixed sampler, overwrite flat-loss checkpoint` |
| 3 | `sprint4.1: update LESSONS_LEARNED — center-crop fg and hallucination` |
| 4 | `sprint4.1: verify and define HOLDOUT_ROI constant` |

---

## Notes

- **Do not proceed to Sprint 5 until the retrain shows decreasing loss.** A checkpoint
  from flat-loss training will produce a blank or random inference mask that demonstrates
  nothing.

- **Training sampler does not need to explicitly exclude the held-out ROI.** Random
  sampling across 1592×400×3000 makes accidental overlap negligible over 3 epochs × 8
  patches. A formal exclusion mask is backlog.

- **The slice_preview.png from Sprint 4 is valid** — it reflects correct label alignment
  and sampler behavior. The problem was only with the loss computation, not the data
  loading or visualization.

The training sampler draws from the full `mito_seg/s2` volume `(1592, 400, 3000)`. To
make Sprint 5 inference a fair demonstration, inference must run on a region that was
not used during training.

**Strategy:** Fix a specific ROI in a different part of the volume from the confirmed
mito-dense region used for training (Z~480–576, Y~80–176, X~2382–2478). Choose a ROI
at a spatially distant location and verify it contains foreground.

## Commit Plan
