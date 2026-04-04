"""
train.py — single entry point for mitotrain.

Sprint 0: all steps call stubs. Replace stubs sprint by sprint per PLANNING.md.
"""

import os
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


# ── Training constants ────────────────────────────────────────────────────────

N_EPOCHS       = 3
BATCH_SIZE     = 2      # patches per gradient step
N_PATCHES      = 8      # patches sampled per epoch
LEARNING_RATE  = 1e-4
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR     = "outputs"

# Held-out region for Sprint 5 inference — never seen during training.
# Training ROI: Z 480-612, Y 80-212, X 2382-2514
# This ROI is ~320 Z-slices away and ~1200 X-voxels away.
HOLDOUT_ROI = (slice(800, 932), slice(80, 212), slice(1200, 1332))

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

# ── Epoch loop ────────────────────────────────────────────────────────────────

for epoch in range(1, N_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{N_EPOCHS}")

    print("  Sampling patches...")
    patches = sample_patches(em_array, seg_array, patch_size=PATCH_SIZE, n_patches=N_PATCHES)
    mean_fg = sum(p[2] for p in patches) / len(patches)
    print(f"  n={len(patches)}, mean_fg: {mean_fg:.2f}")

    epoch_losses = []
    n_batches = N_PATCHES // BATCH_SIZE
    for b in range(n_batches):
        batch = patches[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        x = torch.stack([torch.from_numpy(p[0].astype(np.float32)) for p in batch])
        labels = torch.stack([torch.from_numpy(p[1].astype(np.float32)) for p in batch])

        optimizer.zero_grad()
        out = model(x)
        target = center_crop(labels, tuple(out.shape[2:]))
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        print(f"  Batch {b+1}/{n_batches}  loss: {loss.item():.4f}")

    print(f"  Epoch mean loss: {sum(epoch_losses)/len(epoch_losses):.4f}")

# ── Post-loop ─────────────────────────────────────────────────────────────────

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{N_EPOCHS:03d}.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"\n[Done] Saved checkpoint: {ckpt_path}")

best = max(patches, key=lambda p: p[2])
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = f"{OUTPUT_DIR}/slice_preview.png"
save_slice(best[0], best[1], fg_frac=best[2], path=out_path)
print(f"[Done] Saved visualization: {out_path}")
