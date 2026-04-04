# Sprint 3 — Model Wired Up

**Status:** Ready for implementation.

**Goal:** Replace the identity stub in `model/unet.py` with the real funkelab 3D U-Net.
Confirm that data flows through the full pipeline — from zarr handle to sampled patch to
U-Net forward pass to BCE loss — without shape errors. No training loop yet; this sprint
is purely about getting the shapes right.

**Exit condition:** `python train.py` runs a single forward pass through the real U-Net,
computes a real binary cross-entropy loss against a center-cropped label, and prints a
non-stub loss value. Output shape is `(1, 1, 40, 40, 40)`.

---

## Decisions from Sprint 2 Summary

| Decision | Value | Source |
|----------|-------|--------|
| U-Net input patch size | 132³ | Shape test: smallest valid input |
| U-Net output size | 40³ | Shape test: (1, 1, 40, 40, 40) |
| Label crop strategy | Center-crop 132³ → 40³ | Standard for valid-convolution U-Nets |
| funkelab installed | `funlib.learn.torch 0.1.0` | Already in venv — no reinstall needed |
| Runtime | `.venv/` Python 3.12, zarr 2.18.7 | System Python 3.14 breaks zarr v2 |

---

## Tasks

### Step 0 — Zarr version guard

Add a version assertion near the top of `train.py` to catch environment drift early:

```python
import zarr
assert zarr.__version__.startswith("2."), (
    f"zarr v2 required (N5FSStore); got {zarr.__version__}. "
    "Run inside .venv — do not use system Python."
)
```

This surfaces the environment problem immediately rather than producing a cryptic
`AttributeError: module 'zarr' has no attribute 'N5FSStore'` mid-run.

### Step 1 — Implement `model/unet.py`

Replace the identity stub with the funkelab U-Net. Keep `get_model()` as the public
interface — `train.py` calls `get_model()` and should not need to change.

```python
from funlib.learn.torch.models import UNet

def get_model():
    """
    3D U-Net from funkelab/funlib.learn.torch.
    Input:  (batch, 1, 132, 132, 132) float32
    Output: (batch, 1,  40,  40,  40) float32 (logits, not probabilities)

    Valid convolutions crop the output spatially. The input patch (132³)
    and output patch (40³) are verified by the Sprint 2 shape test.
    Center-crop the label to 40³ before computing loss — do not resize the output.
    """
    return UNet(
        in_channels=1,
        num_fmaps=12,
        fmap_inc_factor=5,
        downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
        kernel_size_down=[[[3, 3, 3], [3, 3, 3]]] * 4,
        kernel_size_up=[[[3, 3, 3], [3, 3, 3]]] * 3,
        num_fmaps_out=1,
        constant_upsample=True,
    )
```

**Parameter count:** print `sum(p.numel() for p in model.parameters())` at instantiation.
Should be in the hundreds of thousands. If 0, the stub is still active.

### Step 2 — Implement center-crop helper

The U-Net output (40³) is smaller than the input patch (132³) due to valid convolutions.
The label patch must be cropped to match before loss is computed. Add a helper to
`utils/` or inline in `train.py`:

```python
def center_crop(label, target_shape):
    """
    Crop label tensor to target_shape, centered.
    label:        torch.Tensor, shape (batch, 1, D, H, W)
    target_shape: tuple (D, H, W) — the U-Net output spatial dims
    Returns:      torch.Tensor, shape (batch, 1, *target_shape)
    """
    starts = [
        (label.shape[i + 2] - target_shape[i]) // 2
        for i in range(3)
    ]
    return label[
        :, :,
        starts[0]:starts[0] + target_shape[0],
        starts[1]:starts[1] + target_shape[1],
        starts[2]:starts[2] + target_shape[2],
    ]
```

For 132³ → 40³: `starts = [(132 - 40) // 2] * 3 = [46, 46, 46]`

### Step 3 — Update `train.py` steps 4–6

Replace stub model call and stub loss with real implementations.

**Step 4 — instantiate model:**
```python
model = get_model()
n_params = sum(p.numel() for p in model.parameters())
print(f"[4/7] Instantiating model...      params: {n_params:,}")
```

**Step 5 — forward pass:**
```python
# Stack patches into a batch tensor: (n_patches, 1, 132, 132, 132)
raw_batch = torch.stack([
    torch.from_numpy(raw).unsqueeze(0) for raw, _, _ in patches
])
output = model(raw_batch)   # → (n_patches, 1, 40, 40, 40)
print(f"[5/7] Forward pass...             output shape: {tuple(output.shape)}")
```

**Step 6 — compute real BCE loss:**
```python
import torch.nn as nn

criterion = nn.BCEWithLogitsLoss()

label_batch = torch.stack([
    torch.from_numpy(label.astype('float32')).unsqueeze(0)
    for _, label, _ in patches
])
label_cropped = center_crop(label_batch, target_shape=(40, 40, 40))

loss = criterion(output, label_cropped)
print(f"[6/7] Computing loss...           loss: {loss.item():.4f}")
```

> Note: `BCEWithLogitsLoss` expects raw logits (not sigmoid-activated output) and float32
> labels in {0.0, 1.0}. The U-Net output is logits by default — do not apply sigmoid
> before the loss.

### Step 4 — Sanity checks

- [ ] `params` printed in step 4 is non-zero (real model, not stub)
- [ ] Output shape is `(n_patches, 1, 40, 40, 40)`
- [ ] Loss is a finite float (not `nan`, not `inf`)
- [ ] Loss is in a plausible range: ~0.5–1.5 for an untrained binary classifier
- [ ] Zarr version assertion fires correctly if run outside the venv
- [ ] No shape mismatch errors between output and cropped label

---

## Commit Plan

| Commit | Message |
|--------|---------|
| 1 | `sprint3: add zarr version guard to train.py` |
| 2 | `sprint3: implement real funkelab U-Net in model/unet.py` |
| 3 | `sprint3: add center_crop helper, wire real loss in train.py` |
| 4 | `sprint3: sanity checks pass, real loss printed` |

---

## Notes and Carry-Forward

- **No training loop yet.** Sprint 3 is one forward pass only. The training loop
  (optimizer, backprop, epoch loop, checkpoint saving) is Sprint 4.

- **Output shape dependency:** The `target_shape=(40, 40, 40)` in `center_crop` is
  hardcoded here based on the Sprint 2 shape test. If the U-Net config changes, this
  must be updated. A cleaner approach (Sprint 4+) is to infer output shape dynamically
  from a dummy forward pass rather than hardcoding it.

- **Batch size:** Sprint 3 uses all 8 sampled patches as a batch. This may be slow on
  CPU — reduce to `n_patches=2` or `n_patches=4` in `train.py` if forward pass takes
  more than ~60 seconds.

- **zarr v3 / N5 (backlog):** The version guard is a stopgap. The proper fix is to
  investigate `n5py` or the zarr v3 N5 compatibility layer before upgrading the Python
  runtime. This is in `TODO_ENHANCEMENTS.md`.
