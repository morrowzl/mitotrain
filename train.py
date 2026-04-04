"""
train.py — single entry point for mitotrain.

Sprint 0: all steps call stubs. Replace stubs sprint by sprint per PLANNING.md.
"""

import numpy as np
import torch
import torch.nn as nn

from data.loader import open_arrays, PATCH_SIZE
from data.sampler import sample_patches
from model.unet import get_model
from utils.visualize import save_slice

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
best = max(patches, key=lambda p: p[2])
out_path = f"{OUTPUT_DIR}/slice_preview.png"
save_slice(best[0], best[1], fg_frac=best[2], path=out_path)
print(f"      {out_path}")

print("Done.")
