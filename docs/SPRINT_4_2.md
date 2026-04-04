# Sprint 4.2 — Extended Training

**Status:** Ready for implementation.

**Goal:** Increase training volume enough to produce a checkpoint that demonstrates
measurable learning — a loss curve that visibly descends over time. This is the minimum
bar for Sprint 5 inference to show anything meaningful.

**Exit condition:** `python train.py` completes with epoch mean loss showing a clear
downward trend from epoch 1 to final epoch. Final epoch mean loss noticeably below 0.693.
New checkpoint saved.

---

## Context from Sprint 4.1

| Metric | Sprint 4.1 value | Problem |
|--------|-----------------|---------|
| N_EPOCHS | 3 | Too few — 12 total gradient steps |
| N_PATCHES | 8 | Too few — small gradient signal per epoch |
| Total gradient steps | 12 | Far too few for 93M-param model |
| Epoch mean loss | ~0.705 (flat) | Gradients flowing but not enough steps |
| Sampler attempts/epoch | 71–96 | High — foreground is sparse in full volume |

Loss above ln(2) (~0.6931) with foreground present indicates the model is attempting
to learn but hasn't had enough updates to move weights meaningfully.

---

## Tasks

### Step 1 — Increase training budget

Update constants in `train.py`:

```python
N_EPOCHS   = 30     # was 3  — enough steps for 93M params to show movement
N_PATCHES  = 16     # was 8  — more signal per epoch, fewer sampler retries wasted
BATCH_SIZE = 2      # unchanged — CPU memory constraint
```

This gives `30 × (16/2) = 240` gradient steps — a minimum viable training run for a
model this size.

> **Wall-clock estimate:** Sprint 4.1 ran 3 epochs in under 10 minutes. 30 epochs at
> similar throughput ≈ ~100 minutes. If this exceeds available time, reduce to
> `N_EPOCHS=15` as a fallback and note in the summary. Do not reduce below 15.

### Step 2 — Add per-epoch loss logging to a file

Rather than relying on console output alone, write epoch losses to a simple text file
so the loss curve survives the session:

```python
# After each epoch, append to loss log
with open("outputs/loss_log.txt", "a") as f:
    f.write(f"epoch {epoch+1}: {epoch_mean_loss:.6f}\n")
```

This gives a persistent record that can be verified independently of the summary doc —
directly addressing the Sprint 4 hallucination issue.

### Step 3 — Add phase timing and resource instrumentation

Add `psutil` to `requirements.txt` if not already present, then instrument `train.py`
with per-phase timing and a resource snapshot function. This is required — not optional.
With 30 epochs and high sampler attempt counts, we need to know where time is going
before declaring the run acceptable.

**Add to `requirements.txt`:**
```
psutil
```

**Add resource snapshot helper near the top of `train.py`:**
```python
import time
import psutil
import os

def log_resources(label):
    proc = psutil.Process(os.getpid())
    mem  = proc.memory_info().rss / 1e9
    cpu  = psutil.cpu_percent(interval=0.1)
    io   = proc.io_counters()
    print(f"  [{label}]  CPU={cpu:.0f}%  MEM={mem:.2f}GB  "
          f"net_read={io.read_bytes/1e6:.0f}MB  write={io.write_bytes/1e6:.0f}MB")
```

**Add phase timing inside the epoch loop:**
```python
for epoch in range(N_EPOCHS):

    t_sample = time.time()
    patches  = sample_patches(...)
    sample_elapsed = time.time() - t_sample

    epoch_losses = []
    forward_total  = 0.0
    backward_total = 0.0

    for i in range(0, N_PATCHES, BATCH_SIZE):
        batch = patches[i:i + BATCH_SIZE]

        t_fwd   = time.time()
        output  = model(raw_batch)
        forward_total += time.time() - t_fwd

        t_bwd   = time.time()
        loss.backward()
        optimizer.step()
        backward_total += time.time() - t_bwd

    print(f"  Timing — sample: {sample_elapsed:.1f}s  "
          f"forward: {forward_total:.1f}s  backward: {backward_total:.1f}s")
    log_resources(f"epoch {epoch+1}")
```

**Also log total wall-clock time at end of run:**
```python
t_total_start = time.time()   # add before epoch loop
# ... training loop ...
total_elapsed = time.time() - t_total_start
print(f"\n[Timing] Total wall-clock: {total_elapsed/60:.1f} minutes")

# Append to loss log
with open("outputs/loss_log.txt", "a") as f:
    f.write(f"total_time_minutes: {total_elapsed/60:.1f}\n")
```

**What to look for in the output:**

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| `sample >> forward + backward` | S3 network fetches are the bottleneck | Pre-compute foreground index (backlog) or reduce patch size |
| `forward >> backward` | Model too large for CPU | Reduce `num_fmaps` |
| `backward >> forward` | Unusual — check for gradient accumulation bug | Verify `optimizer.zero_grad()` placement |
| MEM > 8GB | Memory pressure | Reduce `BATCH_SIZE` or `N_PATCHES` |
| CPU < 50% consistently | GIL contention or I/O waiting | Not much to do on CPU; note for GPU migration |

### Step 4 — Sanity checks

- [ ] `outputs/loss_log.txt` exists and contains one line per epoch plus total time
- [ ] Final epoch loss is lower than epoch 1 loss — **verified from log file, not summary**
- [ ] Loss never goes `nan` or `inf`
- [ ] New checkpoint saved to `checkpoints/epoch_030.pt`
- [ ] Per-epoch timing printed: sample / forward / backward split visible in console
- [ ] Resource snapshot printed per epoch: CPU%, MEM, cumulative read bytes
- [ ] Summary doc includes actual timing numbers from console — not estimated

---

## What to do if loss still doesn't decrease after 30 epochs

In order of likelihood and ease:

1. **Increase learning rate to `1e-3`** — the most likely fix. `1e-4` may be too
   conservative for this model size and batch size.
2. **Reduce `num_fmaps` from 12 to 6** — cuts parameters ~4×, faster convergence per
   step. Changes output shape — re-run shape test and update `OUTPUT_SIZE` if changed.
3. **Increase `N_PATCHES` to 32** — more foreground per epoch.

Try option 1 first before changing model architecture.

---

## Commit Plan

| Commit | Message |
|--------|---------|
| 1 | `sprint4.2: increase N_EPOCHS=30, N_PATCHES=16` |
| 2 | `sprint4.2: add per-epoch loss file logging` |
| 3 | `sprint4.2: add phase timing and psutil resource instrumentation` |
| 4 | `sprint4.2: training complete, loss curve descending` |

---

## Notes

- **Sampler attempt count will increase with N_PATCHES=16.** If attempts regularly
  exceed 200 per epoch, add a note to the summary and flag for the foreground index
  backlog item in `TODO_ENHANCEMENTS.md`.

- **Do not increase BATCH_SIZE.** CPU memory with 93M params and 132³ patches at
  batch size 2 is already substantial. Increasing batch size risks OOM errors.

- **Sprint 5 is one step away.** Once loss_log.txt shows a clear downward trend,
  Sprint 4.2 is done and Sprint 5 inference can proceed immediately.
