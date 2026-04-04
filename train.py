"""
train.py — single entry point for mitotrain.

Sprint 0: all steps call stubs. Replace stubs sprint by sprint per PLANNING.md.
"""

import zarr
import numpy as np
import torch
import torch.nn as nn

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


# ── Config (hardcoded for now; becomes a config file in Sprint 4) ──────────────

N_PATCHES  = 8
OUTPUT_DIR = "outputs"

# ── Pipeline ──────────────────────────────────────────────────────────────────

print("[1/7] Loading subvolume...")
em_array, seg_array = open_arrays()
print(f"      shape: {em_array.shape}")

print("[2/7] Generating binary mask...")
print(f"      deferred to sampler")

print("[3/7] Sampling patches...")
patches = sample_patches(em_array, seg_array, patch_size=PATCH_SIZE, n_patches=N_PATCHES)
raw_patch, label_patch, _ = patches[0]
mean_fg = sum(p[2] for p in patches) / len(patches)
print(f"      n={len(patches)}, fg_frac: {mean_fg:.2f}")

print("[4/7] Instantiating model...")
model = get_model()
n_params = sum(p.numel() for p in model.parameters())
print(f"      params: {n_params}")

print("[5/7] Forward pass...")
x = torch.stack([torch.from_numpy(p[0].astype(np.float32)) for p in patches])  # (N, 1, P, P, P)
with torch.no_grad():
    out = model(x)
print(f"      output shape: {tuple(out.shape)}")

print("[6/7] Computing loss...")
labels = torch.stack([torch.from_numpy(p[1].astype(np.float32)) for p in patches])  # (N, 1, P, P, P)
target = center_crop(labels, (40, 40, 40))  # (N, 1, 40, 40, 40)
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(out, target)
print(f"      loss: {loss.item():.4f}")

print("[7/7] Saving visualization...")
best = max(patches, key=lambda p: p[2])
out_path = f"{OUTPUT_DIR}/slice_preview.png"
save_slice(best[0], best[1], fg_frac=best[2], path=out_path)
print(f"      {out_path}")

print("Done.")
