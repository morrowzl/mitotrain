"""
inspect_binarization.py

Research script for Sprint 2 planning.
Purpose: Determine the true value encoding of mito_pred at s0 and s2, and establish
the correct binarization strategy before writing the Sprint 2 patch sampler.

Background:
- The heinrich-2021a README describes prediction volumes as "a scalar field of uint8
  values that are HIGH inside an object class and LOW outside" — implying soft predictions
  (0–255), not binary {0, 1}.
- Sprint 1 observed values {0, 1, 2, ..., 9} in mito_pred/s2, which is unexpected for
  either a binary or soft single-class prediction.
- Possible explanations:
    (a) mito_pred/s2 is a downsampled soft prediction where values >1 are averaging artifacts
    (b) mito_pred/s2 is a multi-class label array, not a single-class prediction
    (c) mito_pred/s0 is truly binary {0,1} and downsampling introduced intermediate values

Questions this script should answer:
    1. What are the unique values in mito_pred/s0? (binary or 0–255?)
    2. What are the unique values in mito_pred/s2? (same or different?)
    3. If s0 is soft (0–255), what threshold separates foreground from background?
    4. How does mito_pred compare to mito_seg at the same location?
    5. What fraction of voxels are foreground at each threshold?

Usage:
    python inspect_binarization.py > docs/SPRINT_2_BINARIZATION.md
"""

import zarr
import numpy as np

N5_PATH   = "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5"
# Use the same ROI origin confirmed in Sprint 1 (z=576, chunk-aligned, mito-dense)
# Expand slightly for better statistical coverage
ROI_Z = slice(480, 672)   # 192 voxels
ROI_Y = slice(90,  186)   # 96 voxels
ROI_X = slice(2390, 2486) # 96 voxels

def open_group():
    store = zarr.N5FSStore(N5_PATH, anon=True)
    return zarr.open(store, mode="r")


def describe_values(arr, name):
    """Print value distribution statistics for a numpy array."""
    unique, counts = np.unique(arr, return_counts=True)
    total = arr.size
    print(f"\n  [{name}]")
    print(f"    shape:  {arr.shape}")
    print(f"    dtype:  {arr.dtype}")
    print(f"    min/max: {arr.min()} / {arr.max()}")
    print(f"    unique values ({len(unique)} total):")
    for v, c in zip(unique, counts):
        pct = 100 * c / total
        bar = "#" * int(pct * 2)
        print(f"      {v:4d}: {c:8d} voxels ({pct:5.2f}%)  {bar}")


def threshold_analysis(arr, name):
    """Show foreground fraction at a range of thresholds."""
    print(f"\n  [{name}] foreground fraction by threshold:")
    total = arr.size
    for t in [1, 10, 50, 100, 127, 200, 254]:
        fg = (arr >= t).sum()
        print(f"    threshold >= {t:3d}: {fg:8d} / {total} = {100*fg/total:.3f}%")


def main():
    print("=" * 60)
    print("  inspect_binarization.py")
    print("  jrc_hela-2 — mito_pred value encoding")
    print("=" * 60)

    group = open_group()

    # ── 1. mito_pred/s0 ───────────────────────────────────────────
    # s0 is full resolution — load a small crop only
    # Scale ROI to s0 coordinates (s2 → s0: multiply Z by ~4.02, X/Y by 4)
    # Approximate: use a small fixed crop near the Sprint 1 ROI
    print("\n[1/4] Loading mito_pred/s0 (small crop)...")
    s0_roi = (slice(2300, 2400), slice(360, 456), slice(9560, 9656))
    pred_s0 = group["labels/mito_pred/s0"][s0_roi]
    describe_values(pred_s0, "mito_pred/s0")
    if pred_s0.max() > 1:
        threshold_analysis(pred_s0, "mito_pred/s0")

    # ── 2. mito_pred/s2 ───────────────────────────────────────────
    print("\n[2/4] Loading mito_pred/s2 (Sprint 1 ROI)...")
    pred_s2 = group["labels/mito_pred/s2"][ROI_Z, ROI_Y, ROI_X]
    describe_values(pred_s2, "mito_pred/s2")
    if pred_s2.max() > 1:
        threshold_analysis(pred_s2, "mito_pred/s2")

    # ── 3. mito_seg/s2 for comparison ─────────────────────────────
    print("\n[3/4] Loading mito_seg/s2 (same ROI, for comparison)...")
    try:
        seg_s2 = group["labels/mito_seg/s2"][ROI_Z, ROI_Y, ROI_X]
        describe_values(seg_s2, "mito_seg/s2")
    except KeyError:
        print("  mito_seg/s2 not found — trying s4...")
        try:
            # s4 ROI: divide s2 coords by ~2
            seg_s4 = group["labels/mito_seg/s4"][
                slice(240, 336), slice(45, 93), slice(1195, 1243)
            ]
            describe_values(seg_s4, "mito_seg/s4")
        except KeyError:
            print("  mito_seg not available at s2 or s4.")

    # ── 4. Overlap analysis ────────────────────────────────────────
    print("\n[4/4] Overlap: mito_pred/s2 vs mito_seg/s2 (if both loaded)...")
    try:
        pred_binary_1  = (pred_s2 > 0).astype(np.uint8)
        pred_binary_127 = (pred_s2 >= 127).astype(np.uint8)
        seg_binary     = (seg_s2 > 0).astype(np.uint8)
        total = pred_s2.size

        def overlap(a, b, label_a, label_b):
            both = np.logical_and(a, b).sum()
            only_a = np.logical_and(a, ~b.astype(bool)).sum()
            only_b = np.logical_and(~a.astype(bool), b).sum()
            print(f"    {label_a} AND {label_b}: {both} voxels")
            print(f"    only {label_a}:           {only_a} voxels")
            print(f"    only {label_b}:           {only_b} voxels")

        overlap(pred_binary_1,   seg_binary, "pred>0",   "seg>0")
        overlap(pred_binary_127, seg_binary, "pred>=127","seg>0")
    except NameError:
        print("  Skipped — mito_seg not available.")

    print("\n" + "=" * 60)
    print("  Done. Key questions to answer from output:")
    print("  1. Are mito_pred/s0 values binary {0,1} or soft (0-255)?")
    print("  2. Do intermediate values in mito_pred/s2 match GT class IDs (1-9)")
    print("     or are they fractional downsampling artifacts?")
    print("  3. Which threshold (>0 or >=127) best agrees with mito_seg?")
    print("=" * 60)


if __name__ == "__main__":
    main()
