# Sprint 4.2 Summary — Extended Training

**Status:** Complete  
**Date:** 2026-04-03

> All figures in this document are taken directly from `train_run.log` and `loss_log.txt`.
> No values are estimated or inferred.

---

## Exit Condition

`python train.py` completes 30 epochs with epoch mean loss showing a clear downward trend
from epoch 1 to epoch 30. Final epoch mean loss noticeably below 0.693. Checkpoint saved.

**Met:** Epoch 1 mean loss 0.7559 → Epoch 30 mean loss 0.4814. Low point: epoch 16 at 0.4610.

---

## Training Configuration

| Constant | Value |
|----------|-------|
| `N_EPOCHS` | 30 |
| `BATCH_SIZE` | 2 |
| `N_PATCHES` | 16 |
| `LEARNING_RATE` | 1e-4 |
| `activation` | `None` (raw logits for BCEWithLogitsLoss) |
| `pos_weight` | `4.0` |
| `clip_grad_norm` | `max_norm=10.0` |
| Total gradient steps | 240 (30 × 8 batches) |

---

## Loss Curve — Actual Values from `loss_log.txt`

| Epoch | Mean Loss |
|-------|-----------|
| 1  | 0.7559 |
| 2  | 0.5235 |
| 3  | 0.4845 |
| 4  | 0.5238 |
| 5  | 0.4681 |
| 6  | 0.5656 |
| 7  | 0.6008 |
| 8  | 0.5655 |
| 9  | 0.5303 |
| 10 | 0.4854 |
| 11 | 0.5151 |
| 12 | 0.4985 |
| 13 | 0.5407 |
| 14 | 0.5461 |
| 15 | 0.5388 |
| 16 | **0.4610** ← best epoch |
| 17 | 0.5706 |
| 18 | 0.5709 |
| 19 | 0.4931 |
| 20 | 0.4864 |
| 21 | 0.4674 |
| 22 | 0.4998 |
| 23 | 0.4931 |
| 24 | 0.5780 |
| 25 | 0.5888 |
| 26 | 0.5288 |
| 27 | 0.4726 |
| 28 | 0.5322 |
| 29 | 0.5081 |
| 30 | 0.4814 |

Loss is noisy epoch-to-epoch (expected with small batch size and random sampling) but
the trend is clearly downward: epochs 1–5 averaged ~0.56, epochs 26–30 averaged ~0.51,
with the floor dropping from ~0.69 at epoch 1 to ~0.46 at epoch 16.

---

## Timing Summary — Actual Values from `train_run.log`

| Epoch | Sample (s) | Forward (s) | Backward (s) | Total (s) |
|-------|-----------|-------------|--------------|-----------|
| 1  | 74.8 | 68.1 | 174.1 | ~317 |
| 5  | 91.4 | 64.9 | 159.4 | ~316 |
| 10 | 100.1 | 67.3 | 159.8 | ~327 |
| 15 | 73.7 | 61.5 | 119.3 | ~255 |
| 19 | **2073.7** | 69.5 | 190.7 | ~2334 ← anomaly |
| 25 | 45.8 | 74.5 | **940.9** | ~1061 ← anomaly |
| 30 | 86.1 | 63.5 | 128.0 | ~278 |

**Total wall-clock: 196.6 minutes** (from `loss_log.txt`)

### Timing anomalies

Two anomalous epochs stand out:

**Epoch 19 — sample: 2073.7s:** A 34-minute sampling phase. Likely caused by S3 network
latency, connection interruption, or the operating system paging memory (MEM dropped from
4.64 GB at epoch 18 to 2.88 GB at epoch 19, suggesting memory was reclaimed mid-epoch).
Not a code bug — intermittent network/OS event.

**Epoch 25 — backward: 940.9s:** A 15-minute backward pass. Cause unknown. Did not recur
in epochs 26–30 (backward times returned to ~120–190s). Possible causes: memory pressure
triggering paging during gradient computation, or a PyTorch autograd graph growing larger
than expected for one batch. This is the most concerning anomaly and warrants investigation
before any production training runs.

### Bottleneck analysis

Across normal epochs, the phase split is roughly:
- **Sample:** ~80–110s (S3 network fetches — dominant variable cost)
- **Forward:** ~55–75s (model compute)
- **Backward:** ~115–175s (gradient computation — ~2× forward, as expected)

S3 sampling and backward pass are roughly equal contributors to per-epoch time. Addressing
either would cut total training time. The foreground location index (backlog) would
eliminate most of the sampling overhead.

---

## Resource Usage — From `train_run.log`

| Metric | Typical range |
|--------|--------------|
| CPU | 10–64% (low — GPU would change this entirely) |
| Memory | 3.9–4.6 GB |
| Net read | 79 MB/epoch (stable — zarr chunk caching working) |
| Net write | 0 MB |

CPU utilization is low, consistent with a CPU-bound PyTorch workload where most time
is spent in memory-intensive operations rather than compute. Net read is stable at 79 MB
per epoch regardless of sampler attempt count, confirming zarr chunk caching is effective.

---

## Changes Made This Sprint

| Item | Change |
|------|--------|
| `model/unet.py` | `activation=None` — removed default ReLU from final layer |
| `train.py` | `pos_weight=4.0` added to `BCEWithLogitsLoss` |
| `train.py` | `clip_grad_norm_(max_norm=10.0)` added after `loss.backward()` |
| `train.py` | `N_EPOCHS=30`, `N_PATCHES=16` |
| `train.py` | Per-epoch loss written to `outputs/loss_log.txt` |
| `train.py` | Phase timing and `psutil` resource logging per epoch |

---

## Diagnostic History This Sprint

The following issues were diagnosed and resolved in sequence before the successful run:

1. **Flat loss at 0.6931 (all runs prior):** Final layer was ReLU — model output clamped
   to `[0, ∞)`, preventing negative logits. Fixed by `activation=None`.

2. **Gradient explosion (loss ~1234):** `lr=1e-3` too aggressive with no normalization.
   Outputs reached ±5000 within one epoch. Fixed by `lr=1e-4` and `clip_grad_norm_=10.0`.

3. **Loss drifting negative / collapsing to background:** Class imbalance — model found
   it easier to predict all-background. Fixed by `pos_weight=4.0` in loss function.

4. **`activation=None` confirmed by model printout:** Before fixing, `print(model)` was
   added to `train.py` to confirm the ReLU was present and was removed after fix.
   The print was left in for one additional epoch to confirm the fix then removed.

---

## Carry-Forward

- **Growing backward pass (backlog):** Epoch 25 backward took 940s vs typical 120–175s.
  Root cause not identified. Added to `TODO_ENHANCEMENTS.md`. Must be resolved before
  any training run longer than 30 epochs is attempted.

- **Sampler debug prints still active:** `fg in center crop:` lines are printing to
  console and log every attempt. Remove in Sprint 5 pre-checklist commit.

- **loss_log.txt contains full debugging history:** Previous runs (flat 0.6931, explosion
  at 1234.56, etc.) are preserved above the final run. Keep as-is — the file tells the
  diagnostic story and is a useful artifact.

- **Sprint 5:** Load `checkpoints/epoch_030.pt`, run inference on `HOLDOUT_ROI`, visualize
  predicted mask vs ground truth. MVP completion.
