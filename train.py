"""
train.py — single entry point for mitotrain.

Sprint 0: all steps call stubs. Replace stubs sprint by sprint per PLANNING.md.
"""

import os
import time
import psutil
import zarr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

assert zarr.__version__.startswith("2."), (
    f"zarr v2 required (got {zarr.__version__}). Activate the venv: .venv\\Scripts\\activate"
)

from data.loader import open_arrays, PATCH_SIZE
from data.sampler import sample_patches
from model.unet import get_model
from utils.visualize import save_slice

def center_crop(tensor, target_shape):
    """Center-crop spatial dims of (B, C, D, H, W) tensor to target_shape."""
    starts = [(tensor.shape[i + 2] - target_shape[i]) // 2 for i in range(3)]
    return tensor[
        :, :,
        starts[0]:starts[0] + target_shape[0],
        starts[1]:starts[1] + target_shape[1],
        starts[2]:starts[2] + target_shape[2],
    ]


def log_resources(label):
    proc = psutil.Process(os.getpid())
    mem  = proc.memory_info().rss / 1e9
    cpu  = psutil.cpu_percent(interval=0.1)
    io   = proc.io_counters()
    print(f"  [{label}]  CPU={cpu:.0f}%  MEM={mem:.2f}GB  "
          f"net_read={io.read_bytes/1e6:.0f}MB  write={io.write_bytes/1e6:.0f}MB")


# ── Training constants ────────────────────────────────────────────────────────

N_EPOCHS       = 30
BATCH_SIZE     = 2      # patches per gradient step
N_PATCHES      = 16     # patches sampled per epoch
LEARNING_RATE  = 1e-4
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR     = "outputs"

# Held-out region for Sprint 5 inference — never seen during training.
# Training ROI: Z 480-612, Y 80-212, X 2382-2514
# This ROI is ~320 Z-slices away and ~1200 X-voxels away.
HOLDOUT_ROI = (slice(200, 332), slice(80, 212), slice(1500, 1632))

# ── Setup ─────────────────────────────────────────────────────────────────────

print("[Setup] Loading zarr handles...")
em_array, seg_array = open_arrays()
print(f"        shapes: {em_array.shape}")

print("[Setup] Instantiating model...")
model = get_model()
model.train()
n_params = sum(p.numel() for p in model.parameters())
print(f"        params: {n_params:,}")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()
print(f"[Setup] Optimizer: Adam lr={LEARNING_RATE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Epoch loop ────────────────────────────────────────────────────────────────

t_total_start = time.time()
print(f"Model training mode: {model.training}")
for epoch in range(N_EPOCHS):
    print(f"\nEpoch {epoch+1}/{N_EPOCHS}")

    t_sample = time.time()
    print("  Sampling patches...")
    patches = sample_patches(em_array, seg_array, patch_size=PATCH_SIZE, n_patches=N_PATCHES)
    sample_elapsed = time.time() - t_sample
    mean_fg = sum(p[2] for p in patches) / len(patches)
    print(f"  n={len(patches)}, mean_fg: {mean_fg:.2f}")

    epoch_losses = []
    n_batches = N_PATCHES // BATCH_SIZE
    forward_total  = 0.0
    backward_total = 0.0

    for b in range(n_batches):
        batch = patches[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        x = torch.stack([torch.from_numpy(p[0].astype(np.float32)) for p in batch])
        labels = torch.stack([torch.from_numpy(p[1].astype(np.float32)) for p in batch])

        optimizer.zero_grad()

        t_fwd = time.time()
        out = model(x)
        forward_total += time.time() - t_fwd

        target = center_crop(labels, tuple(out.shape[2:]))
        loss = criterion(out, target)

        t_bwd = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        backward_total += time.time() - t_bwd

        epoch_losses.append(loss.item())
        print(f"  Batch {b+1}/{n_batches}  loss: {loss.item():.4f}")

    epoch_mean_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"  Epoch mean loss: {epoch_mean_loss:.4f}")
    print(f"  Timing — sample: {sample_elapsed:.1f}s  "
          f"forward: {forward_total:.1f}s  backward: {backward_total:.1f}s")
    log_resources(f"epoch {epoch+1}")

    with open("outputs/loss_log.txt", "a") as f:
        f.write(f"epoch {epoch+1}: {epoch_mean_loss:.6f}\n")

# ── Post-loop ─────────────────────────────────────────────────────────────────

total_elapsed = time.time() - t_total_start
print(f"\n[Timing] Total wall-clock: {total_elapsed/60:.1f} minutes")

with open("outputs/loss_log.txt", "a") as f:
    f.write(f"total_time_minutes: {total_elapsed/60:.1f}\n")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{N_EPOCHS:03d}.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"\n[Done] Saved checkpoint: {ckpt_path}")

best = max(patches, key=lambda p: p[2])
out_path = f"{OUTPUT_DIR}/slice_preview.png"
save_slice(best[0], best[1], fg_frac=best[2], path=out_path)
print(f"[Done] Saved visualization: {out_path}")
