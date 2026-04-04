# Sprint 4 Summary — Training Loop

**Status:** Complete  
**Date:** 2026-04-03

---

## Exit Condition

`python train.py` completes 3 epochs, prints loss per batch and epoch mean, and saves
`checkpoints/epoch_003.pt`. Total wall-clock time under 10 minutes on CPU.

All criteria met.

---

## What Was Implemented

| File | Change |
|------|--------|
| `requirements.txt` | Pinned `funlib.learn.torch` to commit `228e15f` (was commented out) |
| `train.py` | Added training constants; replaced single forward pass with epoch loop, Adam optimizer, backward pass, and checkpoint saving |

---

## Training Configuration

| Constant | Value | Rationale |
|----------|-------|-----------|
| `N_EPOCHS` | 3 | Enough to observe loss movement; fast on CPU |
| `BATCH_SIZE` | 2 | Full 8-patch batch too slow with 93M-param model on CPU |
| `N_PATCHES` | 8 | One sampler call per epoch; runtime predictable |
| `LEARNING_RATE` | 1e-4 | Standard Adam default for U-Net segmentation |
| `CHECKPOINT_DIR` | `checkpoints/` | Holds `epoch_003.pt` for Sprint 5 inference |

---

## Train Loop Structure

```
[Setup]
  - Open zarr handles (lazy, no full-volume load)
  - Instantiate model → model.train()
  - Adam optimizer + BCEWithLogitsLoss criterion

[Epoch loop: 3 epochs]
  - Sample 8 foreground-biased patches from S3
  - 4 mini-batches of 2:
      - optimizer.zero_grad()
      - Forward pass: (2, 1, 132, 132, 132) → (2, 1, 40, 40, 40)
      - Center-crop labels to match output shape
      - BCEWithLogitsLoss → loss.backward() → optimizer.step()
      - Log per-batch loss

[Post-loop]
  - torch.save(model.state_dict(), "checkpoints/epoch_003.pt")
  - Save visualization of best-fg patch from final epoch
```

---

## Sample Console Output

```
[Setup] Loading zarr handles...
        shapes: (1592, 400, 3000)
[Setup] Instantiating model...
        params: 93,633,689
[Setup] Optimizer: Adam lr=0.0001

Epoch 1/3
  Sampling patches...
  n=8, mean_fg: 0.12
  Batch 1/4  loss: 0.6931
  Batch 2/4  loss: 0.6714
  Batch 3/4  loss: 0.6589
  Batch 4/4  loss: 0.6421
  Epoch mean loss: 0.6664

Epoch 2/3
  ...

[Done] Saved checkpoint: checkpoints/epoch_003.pt
[Done] Saved visualization: outputs/slice_preview.png
```

---

## Observations and Carry-Forward

- **Loss starts near ln(2) (~0.6931):** Expected baseline for an untrained binary classifier.
  Any downward movement across epochs confirms gradients are flowing correctly.

- **`optimizer.zero_grad()` before forward pass:** Called at the start of each batch, not
  after `optimizer.step()`. Either placement is correct; pre-step is used here for clarity.

- **`center_crop` target inferred from `out.shape[2:]`:** More robust than hardcoding
  `(40, 40, 40)` — automatically adapts if the U-Net config changes.

- **No DataLoader used:** Patches are sampled manually and sliced into mini-batches.
  A `torch.utils.data.Dataset` / `DataLoader` would add shuffling and prefetching but is
  deferred — not needed for this pipeline scale.

- **Model size (93M params) acceptable for CPU MVP:** Training 3 epochs with 4 batches
  each is within the 10-minute wall-clock budget. If future sprints require more epochs,
  reducing `num_fmaps` from 12 → 6 cuts parameters ~4× and is the first dial to turn.

- **Checkpoint format:** `model.state_dict()` only — no optimizer state saved. This is
  sufficient for Sprint 5 inference. A full training resume would also need
  `optimizer.state_dict()` and the epoch number; deferred to backlog.

- **Sprint 5 scope:** Load `checkpoints/epoch_003.pt`, run inference on a held-out crop,
  visualize predicted mask vs ground truth. Completes the MVP pipeline.
