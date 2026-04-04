# Sprint 4 — Training Loop

**Status:** Ready for implementation.

**Goal:** Add a minimal training loop — optimizer, backpropagation, epoch iteration, loss
logging, and checkpoint saving. The pipeline now trains.

**Exit condition:** `python train.py` completes N epochs, prints loss per epoch showing a
downward trend (or at minimum a non-increasing trend), and saves a model checkpoint to
`checkpoints/`. Total wall-clock time must be under 10 minutes on CPU for N=3 epochs.

---

## Decisions from Sprint 3 Summary

| Decision | Value | Rationale |
|----------|-------|-----------|
| Optimizer | Adam, lr=1e-4 | Standard default for U-Net segmentation tasks |
| Loss | `BCEWithLogitsLoss` | Already wired in Sprint 3; no change |
| N epochs (MVP) | 3 | Enough to observe loss movement; fast enough on CPU |
| Batch size | 2 | 8-patch batch too slow on CPU with 93M-param model |
| Patches per epoch | 8 | One sampler call per epoch; keeps runtime predictable |
| Checkpoint format | `torch.save(model.state_dict(), path)` | Standard; loadable in Sprint 5 |

> **On model size:** 93M parameters is large for CPU training. Reducing `num_fmaps` from
> 12 to 6 would cut parameters roughly 4×. For MVP purposes, keep the current config and
> accept slow training — the goal is a working loop, not a fast one. If a single epoch
> exceeds 5 minutes, reduce `num_fmaps` as the first adjustment and note it in the summary.

---

## Tasks

### Step 0 — Housekeeping (before any code changes)

- [ ] Pin `funlib.learn.torch` in `requirements.txt`:
  ```
  funlib.learn.torch @ git+https://github.com/funkelab/funlib.learn.torch.git@228e15f
  ```
- [ ] Confirm `.venv` is still on zarr 2.18.7 (`pip show zarr`)
- [ ] Run `python train.py` once to confirm Sprint 3 baseline still passes before changes

### Step 1 — Define constants

Add training constants near the top of `train.py`, clearly grouped and commented.
These will move to `config.py` in a future sprint (currently in backlog):

```python
# ── Training constants ────────────────────────────────────────────
N_EPOCHS       = 3
BATCH_SIZE     = 2      # patches per gradient step
N_PATCHES      = 8      # patches sampled per epoch
LEARNING_RATE  = 1e-4
CHECKPOINT_DIR = "checkpoints"
```

### Step 2 — Restructure `train.py`

The current script runs a single forward pass. Restructure into an epoch loop.

**Sketch of new structure:**

```
[Setup]
  - Open zarr handles          (unchanged)
  - Instantiate model          (unchanged)
  - Instantiate optimizer      (new)
  - Instantiate loss criterion (move from step 6)

[Epoch loop: for epoch in range(N_EPOCHS)]
  - Sample N_PATCHES patches   (sampler call, same as now)
  - Split into mini-batches of BATCH_SIZE
  - For each mini-batch:
      - Forward pass
      - Center-crop labels
      - Compute loss
      - loss.backward()
      - optimizer.step()
      - optimizer.zero_grad()
  - Log mean epoch loss

[Post-loop]
  - Save checkpoint
  - Save visualization (best patch from last epoch)
```

**Console output format:**
```
[Setup] Loading zarr handles...     shapes: (1592, 400, 3000)
[Setup] Instantiating model...      params: 93,633,689
[Setup] Optimizer: Adam lr=1e-4

Epoch 1/3
  Sampling patches...    n=8, mean_fg: 0.11
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

### Step 3 — Implement optimizer and backward pass

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(N_EPOCHS):
    patches = sample_patches(em_array, seg_array, PATCH_SIZE, N_PATCHES)
    epoch_losses = []

    for i in range(0, N_PATCHES, BATCH_SIZE):
        batch = patches[i:i + BATCH_SIZE]

        raw_batch = torch.stack([
            torch.from_numpy(raw).unsqueeze(0) for raw, _, _ in batch
        ])
        label_batch = torch.stack([
            torch.from_numpy(label.astype('float32')).unsqueeze(0)
            for _, label, _ in batch
        ])

        optimizer.zero_grad()
        output = model(raw_batch)
        label_cropped = center_crop(label_batch, target_shape=(40, 40, 40))
        loss = criterion(output, label_cropped)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        print(f"  Batch {i//BATCH_SIZE + 1}/{N_PATCHES//BATCH_SIZE}  loss: {loss.item():.4f}")

    print(f"  Epoch mean loss: {sum(epoch_losses)/len(epoch_losses):.4f}")
```

### Step 4 — Save checkpoint

```python
import os

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{N_EPOCHS:03d}.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"[Done] Saved checkpoint: {ckpt_path}")
```

### Step 5 — Sanity checks

- [ ] Loss decreases or stays flat across epochs — any increase on epoch 1→2 with only
  3 epochs and 8 patches is noise, not a bug; focus on whether it's in the right ballpark
- [ ] Loss never goes `nan` or `inf` (would indicate a normalization or dtype bug)
- [ ] Checkpoint file exists at `checkpoints/epoch_003.pt` after run
- [ ] Wall-clock time for 3 epochs logged and under 10 minutes; if over 5 minutes per
  epoch, reduce `num_fmaps` to 6 and note in summary
- [ ] `model.train()` is called before the loop; `model.eval()` is not needed yet
  (inference is Sprint 5)

---

## Commit Plan

| Commit | Message |
|--------|---------|
| 1 | `sprint4: pin funlib in requirements.txt, confirm baseline` |
| 2 | `sprint4: add training constants to train.py` |
| 3 | `sprint4: restructure train.py into epoch loop with optimizer` |
| 4 | `sprint4: add checkpoint saving` |
| 5 | `sprint4: sanity checks pass, loss curve logged` |

---

## Notes and Carry-Forward

- **DataLoader (backlog):** The current approach calls the sampler once per epoch and
  manually batches the results. A proper `torch.utils.data.Dataset` and `DataLoader`
  would handle shuffling, batching, and prefetching. Deferred — not needed for MVP.

- **Loss curve visualization (backlog):** Plotting epoch loss over time would be useful
  for the README and cover letter. Not needed for MVP exit condition.

- **`num_fmaps` tuning:** If training is unacceptably slow, reducing `num_fmaps` from 12
  to 6 is the first dial. Document the change in the sprint summary if made.

- **Sprint 5 scope:** Load the saved checkpoint, run inference on a held-out crop,
  visualize predicted mask vs ground truth. That completes the MVP.
