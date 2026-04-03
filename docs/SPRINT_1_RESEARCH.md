# Sprint 1 Research — explore.py Output and Analysis

**Run date:** 2026-04-02  
**Command:** `PYTHONUTF8=1 python explore.py`  
**Exit code:** 0

> Note: A first run failed with `UnicodeEncodeError` (cp1252 cannot encode zarr tree box-drawing
> characters). Fixed by setting `PYTHONUTF8=1`. Also note zarr v2 emits a `FutureWarning` about
> `N5FSStore` being deprecated in v3 — harmless for now, but see "Open questions" below.

---

## Key Answers

### Which bucket to use?

Both buckets are accessible anonymously.

| Bucket | Purpose | Use for |
|--------|---------|---------|
| `janelia-cosem-publications` | Ground truth crops (hand-annotated, multi-organelle) | Training labels |
| `janelia-cosem-datasets` | Full-volume EM + automated segmentations/predictions | Raw EM input |

**Decision for Sprint 1:** Load raw EM from `janelia-cosem-datasets` (`em/fibsem-uint16`).
Load training labels from the publications bucket ground truth crops
(`volumes/groundtruth/0003/<crop>/labels/all`).

### What are the ground truth crop names and shapes?

Path: `janelia-cosem-publications/.../volumes/groundtruth/0003/`  
All arrays are `uint64` instance segmentation labels.

> **Bug in explore.py:** The script checked `volumes/labels/0003` — the correct path is
> `volumes/groundtruth/0003`. Fix this before Sprint 1.

| Crop | Shape (Z, Y, X) | Notes |
|------|----------------|-------|
| crop1 | (200, 1000, 1000) | |
| crop3 | (500, 800, 800) | |
| crop4 | (476, 600, 600) | also has centrosome sublabel + masks |
| crop6 | (500, 500, 500) | |
| crop7 | (160, 600, 600) | |
| crop8 | (200, 400, 400) | |
| crop9 | (106, 200, 200) | smallest crop |
| crop13 | (220, 320, 320) | |
| crop14 | (130, 300, 300) | |
| crop15 | (128, 300, 300) | |
| crop16 | (400, 400, 400) | also has nucleolus + ribosomes sublabels |
| crop18 | (220, 400, 400) | |
| crop19 | (110, 300, 300) | |
| crop23 | (500, 500, 500) | |
| crop28 | (400, 400, 400) | |
| crop54 | (400, 400, 400) | |
| crop55 | (400, 400, 400) | |
| crop56 | (400, 400, 400) | |
| crop57 | (400, 400, 400) | |
| crop58 | (400, 400, 400) | |
| crop59 | (400, 400, 400) | |
| crop94 | (400, 400, 400) | |
| crop95 | (400, 400, 400) | |
| crop96 | (400, 400, 400) | |
| crop113 | (500, 1000, 1000) | also has ribosomes sublabel |

The `labels/all` array in each crop is a multi-organelle instance segmentation.
Converting to binary mito mask requires knowing which label IDs correspond to mitochondria —
this needs to be checked (see open questions).

### What downsampled resolution level is practical for CPU?

Raw EM (`em/fibsem-uint16`):

| Level | Shape | Voxel size | Voxels |
|-------|-------|-----------|--------|
| s0 | (6368, 1600, 12000) | 4 × 4 × 5.24 nm | 122B |
| s1 | (3184, 800, 6000) | 8 × 8 × 10.5 nm | 15B |
| s2 | (1592, 400, 3000) | 16 × 16 × 21 nm | 1.9B |
| s3 | (796, 200, 1500) | 32 × 32 × 42 nm | 239M |
| s4 | (398, 100, 750) | 64 × 64 × 84 nm | 30M |

**Recommendation:** Start with **s2** (16 nm/vox). It's small enough for lazy remote access
without full download, mitochondria are still clearly resolvable at 16 nm, and patch sampling
at 64–132 voxels covers a meaningful volume. s4 is very coarse (64 nm) and may lose
fine mito structure; s0/s1 are impractical on CPU without significant cropping.

The ground truth crops in the publications bucket appear to be at **s0 resolution** (4–5 nm/vox)
based on their voxel counts. Aligning GT crops to s2 raw EM will require coordinate scaling.

### Mito label paths (datasets bucket)

> **Bug in explore.py:** The script queried `labels/mito/s0` which does not exist as an array —
> it's a nested group. The actual label names use underscores:

| Path | Shape | dtype | Description |
|------|-------|-------|-------------|
| `labels/mito_seg/s0` | (6368, 1600, 12000) | uint16 | Instance segmentation |
| `labels/mito_seg/s4` | (398, 100, 750) | uint16 | 16× downsampled |
| `labels/mito_pred/s0` | (6368, 1600, 12000) | uint8 | Binary prediction (0/1) |
| `labels/mito_pred/s2` | (1592, 400, 3000) | uint8 | 4× downsampled, chunks (96,96,96) — **Sprint 1 target** |
| `labels/mito_pred/s4` | (398, 100, 750) | uint8 | 16× downsampled |

`mito_pred` is already a binary mask and does not need binarization. It covers the full volume,
making it a good alternative to the sparse GT crops for training — though it is a model
prediction, not ground truth.

### Does fibsem_tools work?

Not tested — `fibsem_tools` is not installed. Given that raw zarr access works cleanly
for both buckets, `fibsem_tools` is likely unnecessary for Sprint 1.

---

## Open Questions for Sprint 1 Planning

1. **GT label encoding:** Do the `labels/all` arrays in the publications bucket use a fixed
   label ID for mitochondria, or is the encoding per-crop? Need to inspect attributes or
   the OpenOrganelle documentation to determine how to extract mito voxels from `uint64` labels.

2. **Coordinate alignment:** The GT crops are at s0 resolution. To pair GT labels with
   downsampled EM (s2), need to determine the crop offsets in world coordinates and map
   to s2 voxel coordinates.

3. **mito_pred vs GT crops:** `mito_pred` covers the full volume and is already binary —
   much simpler to use for Sprint 1. The trade-off is it's a model output, not hand-annotated.
   Decision: use `mito_pred/s2` for Sprint 1 (simpler), switch to GT crops for evaluation.

4. **explore.py bugs to fix before Sprint 1:**
   - Path `volumes/labels/0003` → `volumes/groundtruth/0003` (publications bucket)
   - Path `labels/mito/s0` → `labels/mito_seg/s0` or `labels/mito_pred/s0` (datasets bucket)
   - Add `PYTHONUTF8=1` to the usage instructions (or encode tree output as ASCII)

---

## Raw EM Attributes (s0, datasets bucket)

```
path:   em/fibsem-uint16/s0
shape:  (6368, 1600, 12000)
dtype:  uint16
chunks: (64, 64, 64)
attrs:
  pixelResolution: {dimensions: [4.0, 4.0, 5.24], unit: "nm"}
  transform:
    axes:      [z, y, x]
    scale:     [5.24, 4.0, 4.0]
    translate: [0.0, 0.0, 0.0]
    units:     [nm, nm, nm]
```
