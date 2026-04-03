"""
inspect_roi.py

Research script for Sprint 2 planning.
Purpose: Determine whether the mito_pred/s2 labels are spatially aligned with the
visually apparent mitochondria in the raw EM at the Sprint 1 ROI, and find a better
ROI origin if needed.

Background:
- Sprint 1 slice_preview.png showed large oval mitochondria cross-sections visible in
  the raw EM, but the red mito_pred overlay appeared only at the bottom edge of the crop
  (cell boundary region), not overlapping the visible organelles.
- This may indicate:
    (a) The ROI origin is positioned at the cell boundary — mitochondria in the EM are
        just inside the cell, but the label-dense region is elsewhere in Z/Y/X
    (b) mito_pred at 16 nm resolution does not label the internal mitochondria visible
        in this region (label quality issue)
    (c) There is a spatial offset between EM and mito_pred arrays at s2

Questions this script should answer:
    1. Where in the full s2 volume is mito_pred label density highest?
    2. Does the label-dense region correspond visually to mitochondria in the EM?
    3. Is there a better ROI origin that places the crop over a high-density mito region?
    4. Does the EM at the current ROI origin look like cell interior or cell boundary?

Usage:
    python inspect_roi.py
    → saves outputs/inspect_roi_density.png  (Z density profile + best-slice overlay)
    → saves outputs/inspect_roi_grid.png     (grid of Z slices showing EM + overlay)
    → prints recommended ROI origin for Sprint 2
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt

N5_PATH   = "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5"
EM_LAYER  = "em/fibsem-uint16/s2"
SEG_LAYER = "labels/mito_pred/s2"

# Broad survey region — wider than Sprint 1 ROI to find label-dense areas
# Keep Y and X near the neuroglancer-confirmed mito region; expand Z broadly
SURVEY_Z = slice(400, 800)    # 400 Z slices — broad sweep
SURVEY_Y = slice(80,  200)    # 120 Y voxels
SURVEY_X = slice(2350, 2510)  # 160 X voxels


def open_group():
    store = zarr.N5FSStore(N5_PATH, anon=True)
    return zarr.open(store, mode="r")


def normalize_em(arr):
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    return np.clip((arr - p1) / (p99 - p1), 0, 1)


def make_overlay(em_slice, seg_slice):
    """Return an RGBA overlay: red where seg>0, transparent elsewhere."""
    rgba = np.zeros((*seg_slice.shape, 4), dtype=np.float32)
    mask = seg_slice > 0
    rgba[mask] = [1.0, 0.0, 0.0, 0.6]
    return rgba


def main():
    print("Opening N5 store...")
    group = open_group()

    # ── 1. Load survey region ─────────────────────────────────────
    print(f"Loading survey region: Z={SURVEY_Z}, Y={SURVEY_Y}, X={SURVEY_X}")
    print("  (this may take 30–60 seconds for remote access)")

    em_survey  = group[EM_LAYER][SURVEY_Z, SURVEY_Y, SURVEY_X]
    seg_survey = group[SEG_LAYER][SURVEY_Z, SURVEY_Y, SURVEY_X]
    seg_binary = (seg_survey > 0).astype(np.uint8)

    print(f"  EM shape:  {em_survey.shape}, dtype: {em_survey.dtype}")
    print(f"  Seg shape: {seg_survey.shape}, unique values: {np.unique(seg_survey)}")

    # ── 2. Z density profile ──────────────────────────────────────
    z_counts = seg_binary.sum(axis=(1, 2))
    z_abs    = np.arange(SURVEY_Z.start, SURVEY_Z.stop)
    total_voxels = seg_binary.shape[1] * seg_binary.shape[2]

    print("\nMito voxel density by Z slice (top 20):")
    top20_idx = np.argsort(z_counts)[-20:][::-1]
    for i in top20_idx:
        pct = 100 * z_counts[i] / total_voxels
        bar = "#" * int(pct * 3)
        print(f"  z={z_abs[i]:4d}  {z_counts[i]:5d} voxels ({pct:5.2f}%)  {bar}")

    best_local_z = np.argmax(z_counts)
    best_abs_z   = z_abs[best_local_z]
    print(f"\nBest Z slice: z={best_abs_z} ({z_counts[best_local_z]} mito voxels, "
          f"{100*z_counts[best_local_z]/total_voxels:.2f}%)")

    # ── 3. Density profile plot ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Z density bar chart
    axes[0].bar(z_abs, z_counts, color="steelblue", width=1)
    axes[0].axvline(best_abs_z, color="red", linestyle="--", label=f"best z={best_abs_z}")
    axes[0].axvline(576, color="orange", linestyle="--", label="Sprint 1 z=576")
    axes[0].set_xlabel("Z (s2 voxels)")
    axes[0].set_ylabel("Mito voxels in slice")
    axes[0].set_title("Mito label density across Z")
    axes[0].legend(fontsize=8)

    # Panel 2: Raw EM at best Z
    em_best  = normalize_em(em_survey[best_local_z])
    seg_best = seg_binary[best_local_z]
    axes[1].imshow(em_best, cmap="gray")
    axes[1].imshow(make_overlay(em_best, seg_best))
    axes[1].set_title(f"Best Z slice: z={best_abs_z}\n({z_counts[best_local_z]} mito voxels)")
    axes[1].axis("off")

    # Panel 3: Raw EM at Sprint 1 ROI z=576
    sprint1_local = 576 - SURVEY_Z.start
    if 0 <= sprint1_local < em_survey.shape[0]:
        em_s1  = normalize_em(em_survey[sprint1_local])
        seg_s1 = seg_binary[sprint1_local]
        axes[2].imshow(em_s1, cmap="gray")
        axes[2].imshow(make_overlay(em_s1, seg_s1))
        axes[2].set_title(f"Sprint 1 ROI: z=576\n({z_counts[sprint1_local]} mito voxels)")
    else:
        axes[2].text(0.5, 0.5, "z=576 outside survey range",
                     ha="center", va="center", transform=axes[2].transAxes)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/inspect_roi_density.png", dpi=150)
    print("\nSaved outputs/inspect_roi_density.png")

    # ── 4. Grid of slices around the best Z ───────────────────────
    # Show 9 slices centered on best Z to see spatial context
    half = 4
    z_indices = range(
        max(0, best_local_z - half * 4),
        min(em_survey.shape[0], best_local_z + half * 4),
        4  # every 4th slice
    )
    z_indices = list(z_indices)[:9]  # cap at 9 panels

    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))
    for ax, zi in zip(axes2.flat, z_indices):
        em_s  = normalize_em(em_survey[zi])
        seg_s = seg_binary[zi]
        ax.imshow(em_s, cmap="gray")
        ax.imshow(make_overlay(em_s, seg_s))
        ax.set_title(f"z={z_abs[zi]}  ({seg_binary[zi].sum()} mito px)", fontsize=9)
        ax.axis("off")
    for ax in axes2.flat[len(z_indices):]:
        ax.axis("off")

    plt.suptitle("EM + mito_pred overlay — slices around best Z", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/inspect_roi_grid.png", dpi=150)
    print("Saved outputs/inspect_roi_grid.png")

    # ── 5. Recommended ROI for Sprint 2 ───────────────────────────
    # Find the chunk-aligned Z origin closest to best_abs_z
    chunk_z = 96
    recommended_z = (best_abs_z // chunk_z) * chunk_z
    print("\n" + "=" * 60)
    print("  Recommended ROI origin for Sprint 2 (chunk-aligned):")
    print(f"    Z: slice({recommended_z}, {recommended_z + chunk_z})")
    print(f"    Y: slice({SURVEY_Y.start}, {SURVEY_Y.start + 96})")
    print(f"    X: slice({SURVEY_X.start + 32}, {SURVEY_X.start + 128})")
    print(f"  (verify visually in inspect_roi_density.png and inspect_roi_grid.png)")
    print("=" * 60)


if __name__ == "__main__":
    main()
