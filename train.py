"""
train.py — single entry point for mitotrain.

Sprint 0: all steps call stubs. Replace stubs sprint by sprint per PLANNING.md.
"""

import numpy as np
import torch
import torch.nn as nn

from data.loader import load_subvolume
from data.sampler import sample_patches
from model.unet import get_model
from utils.visualize import save_slice

# ── Config (hardcoded for Sprint 0; becomes a config file in Sprint 4) ────────

DATASET    = "jrc_hela-2"
PATCH_SIZE = 32
N_PATCHES  = 8
OUTPUT_DIR = "outputs"

# ── Pipeline ──────────────────────────────────────────────────────────────────

print("[1/7] Loading subvolume...")
raw    = load_subvolume(DATASET, "raw",         roi=None)
labels = load_subvolume(DATASET, "labels/mito", roi=None)
print(f"      shape: {raw.shape}")

print("[2/7] Generating binary mask...")
mask = (labels > 0).astype(np.uint8)
mito_voxels = int(mask.sum())
print(f"      mito voxels: {mito_voxels} (stub)")

print("[3/7] Sampling patches...")
patches = sample_patches(raw, mask, patch_size=PATCH_SIZE, n_patches=N_PATCHES)
raw_patch, label_patch = patches[0]
print(f"      n={len(patches)}, shape: {raw_patch.shape}")

print("[4/7] Instantiating model...")
model = get_model()
n_params = sum(p.numel() for p in model.parameters())
print(f"      params: {n_params} (stub)")

print("[5/7] Forward pass...")
x = torch.from_numpy(raw_patch.astype(np.float32)).unsqueeze(0)  # (1, 1, P, P, P)
with torch.no_grad():
    out = model(x)
print(f"      output shape: {tuple(out.shape)}")

print("[6/7] Computing loss...")
target = torch.from_numpy(label_patch.astype(np.float32)).unsqueeze(0)
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(out, target)
print(f"      loss: {loss.item():.4f} (stub)")

print("[7/7] Saving visualization...")
out_path = f"{OUTPUT_DIR}/slice_preview.png"
save_slice(raw_patch, label_patch, path=out_path)
print(f"      {out_path}")

print("Done.")
