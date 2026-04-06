"""
predict.py — Sprint 5 inference script.
Loads trained checkpoint, runs inference on held-out region,
saves side-by-side visualization.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from data.loader import open_arrays
from model.unet import get_model

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = "checkpoints/epoch_030.pt"
OUTPUT_PATH = "outputs/inference_preview.png"

# Held-out region — never seen during training
HOLDOUT_ROI = (slice(200, 332), slice(80, 212), slice(1500, 1632))
THRESHOLD   = 0.30


def center_crop(tensor, target_shape):
    """Center-crop spatial dims of (B, C, D, H, W) tensor to target_shape."""
    starts = [(tensor.shape[i + 2] - target_shape[i]) // 2 for i in range(3)]
    return tensor[
        :, :,
        starts[0]:starts[0] + target_shape[0],
        starts[1]:starts[1] + target_shape[1],
        starts[2]:starts[2] + target_shape[2],
    ]


def main():
    # ── Load model ────────────────────────────────────────────────────────────────
    print(f"[Setup] Loading checkpoint: {CHECKPOINT}")
    model = get_model()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    print("        model.eval() set")

    # ── Load data ─────────────────────────────────────────────────────────────────
    print("[Setup] Opening zarr arrays...")
    em_array, seg_array = open_arrays()

    print("[Inference] Loading holdout crop...")
    raw_crop = em_array[HOLDOUT_ROI].astype(np.float32)
    p1, p99  = np.percentile(raw_crop, [1, 99])
    raw_norm = np.clip((raw_crop - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    seg_crop  = seg_array[HOLDOUT_ROI]
    gt_binary = (seg_crop > 0).astype(np.uint8)

    # ── Forward pass ──────────────────────────────────────────────────────────────
    x = torch.from_numpy(raw_norm[np.newaxis, np.newaxis]).float()  # (1, 1, 132, 132, 132)
    print(f"[Inference] Input shape: {tuple(x.shape)}")

    with torch.no_grad():
        logits = model(x)                                    # (1, 1, 40, 40, 40)

    print(f"[Inference] Output shape: {tuple(logits.shape)}")
    probs = torch.sigmoid(logits).squeeze().numpy()          # (40, 40, 40)
    pred  = (probs > THRESHOLD).astype(np.uint8)             # (40, 40, 40)

    # ── Align ground truth to output shape ────────────────────────────────────────
    gt_tensor   = torch.from_numpy(gt_binary[np.newaxis, np.newaxis].astype(np.float32))
    gt_cropped  = center_crop(gt_tensor, (40, 40, 40)).squeeze().numpy().astype(np.uint8)

    raw_tensor  = torch.from_numpy(raw_norm[np.newaxis, np.newaxis])
    raw_cropped = center_crop(raw_tensor, (40, 40, 40)).squeeze().numpy()

    # ── Visualize ─────────────────────────────────────────────────────────────────
    best_z = int(np.argmax(gt_cropped.sum(axis=(1, 2))))

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: Raw EM
    axes[0].imshow(raw_cropped[best_z], cmap="gray")
    axes[0].set_title(f"Raw EM — z={best_z}")
    axes[0].axis("off")

    # Panel 2: Ground truth
    axes[1].imshow(raw_cropped[best_z], cmap="gray")
    rgba_gt = np.zeros((*gt_cropped[best_z].shape, 4))
    rgba_gt[gt_cropped[best_z] > 0] = [0, 1, 0, 0.5]   # green = ground truth
    axes[1].imshow(rgba_gt)
    axes[1].set_title(f"Ground truth — z={best_z}\n({gt_cropped[best_z].sum()} fg voxels)")
    axes[1].axis("off")

    # Panel 3: Prediction
    axes[2].imshow(raw_cropped[best_z], cmap="gray")
    rgba_pred = np.zeros((*pred[best_z].shape, 4))
    rgba_pred[pred[best_z] > 0] = [1, 0, 0, 0.5]        # red = prediction
    axes[2].imshow(rgba_pred)
    fg_pct = 100 * pred[best_z].mean()
    axes[2].set_title(f"Prediction (thresh={THRESHOLD}) — z={best_z}\n({pred[best_z].sum()} fg voxels, {fg_pct:.1f}%)")
    axes[2].axis("off")

    # Panel 4 — raw probability map (contrast stretched to actual range)
    prob_slice = probs[best_z]
    axes[3].imshow(prob_slice, cmap="hot", vmin=prob_slice.min(), vmax=prob_slice.max())
    axes[3].set_title(f"Prob map (contrast stretched) — z={best_z}\nrange: {prob_slice.min():.3f}–{prob_slice.max():.3f}")
    axes[3].axis("off")

    plt.suptitle(f"Inference on held-out region | Checkpoint: {CHECKPOINT}", fontsize=11)
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\n[Done] Saved: {OUTPUT_PATH}")

    # ── Statistics ────────────────────────────────────────────────────────────────
    print(f"\nInference statistics:")
    print(f"  Holdout ROI:        {HOLDOUT_ROI}")
    print(f"  Output shape:       {pred.shape}")
    print(f"  GT fg voxels:       {gt_cropped.sum()} / {gt_cropped.size} ({100*gt_cropped.mean():.2f}%)")
    print(f"  Pred fg voxels:     {pred.sum()} / {pred.size} ({100*pred.mean():.2f}%)")
    print(f"  Logit min/max:      {logits.min():.4f} / {logits.max():.4f}")
    print(f"  Prob min/max:       {probs.min():.4f} / {probs.max():.4f}")


if __name__ == "__main__":
    main()
