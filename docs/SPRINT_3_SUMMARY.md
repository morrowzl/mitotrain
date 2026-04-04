# Sprint 3 Summary — Real U-Net Forward Pass

**Status:** Complete  
**Date:** 2026-04-03

---

## Exit Condition

`python train.py` runs a single forward pass through the real funkelab 3D U-Net, computes a
real binary cross-entropy loss against a center-cropped label tensor, and prints a finite,
non-stub loss value. Output shape is `(8, 1, 40, 40, 40)`.

All criteria met. Loss: 0.6931 (~ln(2), expected for an untrained binary classifier).

---

## What Was Implemented

| File | Change |
|------|--------|
| `model/unet.py` | Replaced `IdentityUNet` stub with real funkelab `UNet`; `get_model()` kept as public interface; `IdentityUNet` class removed entirely |
| `train.py` | Added zarr v2 version guard (assert fires outside `.venv`); added `center_crop()` helper; step 5 now stacks all 8 patches into one batch before forward pass; step 6 center-crops labels to 40³ and computes real `BCEWithLogitsLoss` |

---

## U-Net Configuration

```python
UNet(
    in_channels=1,
    num_fmaps=12,
    fmap_inc_factor=5,
    downsample_factors=[(2,2,2), (2,2,2), (2,2,2)],
    kernel_size_down=[[[3,3,3],[3,3,3]]] * 4,
    kernel_size_up=[[[3,3,3],[3,3,3]]] * 3,
    num_fmaps_out=1,
    constant_upsample=True,
)
```

- **Param count:** 93,633,689
- **Input:** `(N, 1, 132, 132, 132)` → **Output:** `(N, 1, 40, 40, 40)`
- Valid convolutions throughout — output is spatially smaller than input by design

---

## Center-Crop Strategy

Label patches (132³) are center-cropped to match U-Net output (40³) before loss computation:

```
starts = [(132 - 40) // 2] * 3 = [46, 46, 46]
```

This is the standard approach for valid-convolution U-Nets: the loss is computed only over the
central region where receptive fields are fully populated.

---

## Installation Note

`funlib.learn.torch` was not present in `.venv` at sprint start. Installed during sprint:

```
pip install git+https://github.com/funkelab/funlib.learn.torch.git
```

Resolved to commit `228e15f` (v0.1.0). `requirements.txt` should be updated to pin this.

---

## Runtime Sanity Check Results

```
[1/7] Loading subvolume...        shape: (1592, 400, 3000)
[2/7] Generating binary mask...   deferred to sampler
[3/7] Sampling patches...         n=8, fg_frac: 0.14
[4/7] Instantiating model...      params: 93633689
[5/7] Forward pass...             output shape: (8, 1, 40, 40, 40)
[6/7] Computing loss...           loss: 0.6931
[7/7] Saving visualization...     outputs/slice_preview.png
```

---

## Observations and Carry-Forward

- **Loss is ~ln(2):** This is the expected BCE loss for an untrained classifier outputting
  near-zero logits across a balanced (or near-balanced) label distribution. Confirms the forward
  pass is wired correctly.

- **Batch size = N_PATCHES = 8:** The current forward pass batches all sampled patches in one
  call. Sprint 4 training loop will need a proper DataLoader and configurable batch size — the
  hardcoded `N_PATCHES=8` is a placeholder.

- **93M parameters is large for a first training run:** The U-Net config (12 fmaps,
  fmap_inc_factor=5, 3 levels) produces a large model. If training is slow or GPU memory is
  tight in Sprint 4, consider reducing `num_fmaps` or `fmap_inc_factor` as the first dial to
  turn. The architecture is otherwise correct for 3D mitochondria segmentation.

- **`requirements.txt` not yet updated:** `funlib.learn.torch` is installed in `.venv` but not
  pinned in `requirements.txt`. Add before Sprint 4 so the environment can be reproduced.

- **zarr v3 / N5 (backlog, still open):** `N5FSStore` deprecation warning persists. The zarr
  guard added this sprint will surface the issue clearly if the venv is accidentally skipped.
  Investigate `n5py` as the migration path before upgrading the Python runtime beyond 3.12.

- **Sprint 4 scope:** Training loop — optimizer, learning rate, checkpoint saving, loss curve
  logging. The full data pipeline is now in place; the remaining work is the training harness.
