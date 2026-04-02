"""
explore.py

Sprint 0 data exploration script for mitotrain.
Purpose: verify S3 access, understand the N5 group structure for jrc_hela-2,
and surface the information needed to write Sprint 1's planning doc.

Run this before writing any pipeline code. Results should be pasted into
SPRINT_1_RESEARCH.md or equivalent notes.

Usage:
    pip install "fsspec[s3]" zarr dask fibsem_tools
    python explore.py
"""

import zarr
import s3fs
import json

# ── Config ────────────────────────────────────────────────────────────────────

DATASET = "jrc_hela-2"

# Two buckets — we check both
PUBLICATIONS_ROOT = f"s3://janelia-cosem-publications/heinrich-2021a/{DATASET}/{DATASET}.n5"
DATASETS_ROOT     = f"s3://janelia-cosem-datasets/{DATASET}/{DATASET}.n5"

# ── Helpers ───────────────────────────────────────────────────────────────────

def open_n5(s3_path):
    """Open an N5 store anonymously and return the root zarr group."""
    store = zarr.N5FSStore(s3_path, anon=True)
    return zarr.open(store, mode="r")


def print_tree(group, label):
    """Print the full array/group tree for a zarr group."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {group.store.path}")
    print(f"{'='*60}")
    print(group.tree())


def describe_array(group, path, label):
    """Print shape, dtype, chunk size, and voxel count for a named array."""
    try:
        arr = group[path]
        voxels = 1
        for dim in arr.shape:
            voxels *= dim
        print(f"\n  [{label}]")
        print(f"    path:   {path}")
        print(f"    shape:  {arr.shape}")
        print(f"    dtype:  {arr.dtype}")
        print(f"    chunks: {arr.chunks}")
        print(f"    voxels: {voxels:,}")
        if arr.attrs:
            print(f"    attrs:  {json.dumps(dict(arr.attrs), indent=6)}")
    except KeyError:
        print(f"\n  [{label}] NOT FOUND at path: {path}")


def list_group(group, path, label):
    """List the immediate children of a group."""
    try:
        g = group[path]
        children = list(g.keys())
        print(f"\n  [{label}] children of '{path}':")
        for c in children:
            print(f"    - {c}")
        return children
    except KeyError:
        print(f"\n  [{label}] group NOT FOUND at path: {path}")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # ── 1. Publications bucket (paper data / ground truth) ──────────────────
    print("\n[1/4] Opening publications bucket...")
    try:
        pub = open_n5(PUBLICATIONS_ROOT)
        print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        pub = None

    if pub:
        # Full tree — may be large; comment out if too verbose
        print_tree(pub, f"Publications bucket: {DATASET}")

        # Raw EM
        describe_array(pub, "volumes/raw/s0", "Raw EM s0 (full res)")
        describe_array(pub, "volumes/raw/s1", "Raw EM s1 (half res)")
        describe_array(pub, "volumes/raw/s2", "Raw EM s2 (quarter res)")

        # Ground truth crop listing
        crop_names = list_group(pub, "volumes/groundtruth/0003", "Ground truth crops")

        # Describe each crop found
        for crop in crop_names:
            describe_array(
                pub,
                f"volumes/groundtruth/0003/{crop}/labels/all",
                f"GT crop: {crop}"
            )

    # ── 2. Datasets bucket (canonical / segmentations) ───────────────────────
    print("\n[2/4] Opening datasets bucket...")
    try:
        ds = open_n5(DATASETS_ROOT)
        print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        ds = None

    if ds:
        print_tree(ds, f"Datasets bucket: {DATASET}")

        # Raw EM
        describe_array(ds, "em/fibsem-uint16/s0", "Raw EM s0 (full res)")
        describe_array(ds, "em/fibsem-uint16/s4", "Raw EM s4 (16x downsampled)")

        # Predictions — mito
        list_group(ds, "labels", "Labels group")
        describe_array(ds, "labels/mito_seg/s0", "Mito instance seg s0")
        describe_array(ds, "labels/mito_seg/s4", "Mito instance seg s4 (16x downsampled)")
        describe_array(ds, "labels/mito_pred/s0", "Mito binary pred s0")
        describe_array(ds, "labels/mito_pred/s4", "Mito binary pred s4 (16x downsampled)")

    # ── 3. fibsem_tools access check ─────────────────────────────────────────
    print("\n[3/4] Checking fibsem_tools...")
    try:
        from fibsem_tools import read
        creds = {"anon": True}
        g = read(DATASETS_ROOT + "/em/fibsem-uint16/", storage_options=creds)
        print(f"  fibsem_tools OK — arrays: {tuple(k for k, _ in g.arrays())}")
    except ImportError:
        print("  fibsem_tools not installed — run: pip install fibsem_tools")
    except Exception as e:
        print(f"  fibsem_tools FAILED: {e}")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("\n[4/4] Done. Paste output into SPRINT_1_RESEARCH.md.")
    print("  Key questions this should answer:")
    print("  - Which bucket to use for training labels?")
    print("  - What are the ground truth crop names and shapes?")
    print("  - What downsampled resolution level is practical for patch sampling?")
    print("  - Does fibsem_tools open both buckets cleanly?")


if __name__ == "__main__":
    main()
