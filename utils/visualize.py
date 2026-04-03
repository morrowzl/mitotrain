import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_slice(raw, labels, path):
    """
    Save a side-by-side visualization of a raw EM slice and binary mito mask.

    Args:
        raw:    np.ndarray of shape (1, Z, Y, X), float32 in [0, 1]
        labels: np.ndarray of shape (1, Z, Y, X), uint8 binary mask
        path:   str — output file path
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Pick the Z slice with the most mito voxels (middle slice is often empty)
    mito_per_z = labels[0].sum(axis=(1, 2))  # (Z,)
    z_best = int(mito_per_z.argmax())
    raw_slice   = raw[0, z_best]      # (Y, X), float32
    label_slice = labels[0, z_best]   # (Y, X), uint8

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(raw_slice, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Raw EM — z={z_best}")
    axes[0].axis("off")

    # Build RGBA overlay: red channel where mask=1, alpha=0.55
    rgba = np.zeros((*label_slice.shape, 4), dtype=np.float32)
    rgba[..., 0] = 1.0                    # red
    rgba[..., 3] = label_slice * 0.55    # alpha: 0 where no mito, 0.55 where mito

    axes[1].imshow(raw_slice, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].imshow(rgba, interpolation="nearest")
    axes[1].set_title(f"Mito mask overlay — z={z_best}")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
