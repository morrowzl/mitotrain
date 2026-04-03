# inspect_mito_seg.py
import zarr
import numpy as np
import matplotlib.pyplot as plt

N5_PATH   = "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5"
EM_LAYER  = "em/fibsem-uint16/s2"
SEG_LAYER = "labels/mito_seg/s2"  # switching from mito_pred to mito_seg

# Use the known mito-dense Z range from inspect_roi findings
ROI_Z = slice(480, 576)
ROI_Y = slice(80,  176)
ROI_X = slice(2382, 2478)

store = zarr.N5FSStore(N5_PATH, anon=True)
group = zarr.open(store, mode="r")

# First check mito_seg/s2 even exists
print("Available mito_seg levels:", list(group["labels/mito_seg"].keys()))

seg = group[SEG_LAYER][ROI_Z, ROI_Y, ROI_X]
em  = group[EM_LAYER][ROI_Z, ROI_Y, ROI_X]

# Value distribution
unique, counts = np.unique(seg, return_counts=True)
print(f"\nmito_seg/s2 unique values ({len(unique)} total): {unique[:20]}")
print(f"dtype: {seg.dtype}")
print(f"foreground voxels (> 0): {(seg > 0).sum()} / {seg.size} = {100*(seg>0).mean():.2f}%")

# Binarize
seg_binary = (seg > 0).astype(np.uint8)

# Find best Z slice
z_counts  = seg_binary.sum(axis=(1, 2))
best_z    = np.argmax(z_counts)
print(f"\nBest Z slice: local={best_z}, absolute={ROI_Z.start + best_z} ({z_counts[best_z]} mito voxels)")

# Normalize EM
em_slice  = em[best_z].astype(np.float32)
p1, p99   = np.percentile(em_slice, [1, 99])
em_norm   = np.clip((em_slice - p1) / (p99 - p1), 0, 1)

# Plot: 3 panels — EM only, overlay, seg only
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(em_norm, cmap="gray")
axes[0].set_title(f"Raw EM — z={ROI_Z.start + best_z}")
axes[0].axis("off")

axes[1].imshow(em_norm, cmap="gray")
rgba = np.zeros((*seg_binary[best_z].shape, 4))
rgba[seg_binary[best_z] > 0] = [1, 0, 0, 0.5]
axes[1].imshow(rgba)
axes[1].set_title(f"mito_seg overlay — z={ROI_Z.start + best_z}\n({z_counts[best_z]} mito voxels)")
axes[1].axis("off")

axes[2].imshow(seg_binary[best_z], cmap="gray")
axes[2].set_title("Binary seg mask only")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("outputs/inspect_mito_seg.png", dpi=150)
print("Saved outputs/inspect_mito_seg.png")