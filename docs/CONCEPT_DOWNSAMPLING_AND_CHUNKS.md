# Concept Note: Downsampling, Chunks, and Why Some Numbers Get Larger

---

## What Downsampling Means Physically

The FIB-SEM microscope produces an image where each voxel represents a tiny cube of real
cell. At full resolution (s0), each voxel is approximately 4 × 4 × 5.24 nanometers. That
is smaller than most proteins.

Downsampling produces lower-resolution versions of the same volume by merging groups of
neighboring voxels into single larger voxels. At s2 (two downsampling steps), each voxel
represents roughly 16 × 16 × 21 nm — about 4× larger in each dimension than s0.

Think of it like a photograph:

- **s0** — the original full-resolution image, every pixel sharp
- **s1** — half resolution: half as many pixels in each direction, one quarter the total count
- **s2** — quarter resolution: one sixteenth the total voxels of s0
- **s4** — 1/256th the total voxels of s0

The full jrc_hela-2 volume illustrates this:

| Level | Shape (Z, Y, X) | Voxels | Voxel size |
|-------|----------------|--------|------------|
| s0 | (6368, 1600, 12000) | ~122 billion | 4 × 4 × 5.24 nm |
| s2 | (1592, 400, 3000) | ~1.9 billion | 16 × 16 × 21 nm |
| s4 | (398, 100, 750) | ~30 million | 64 × 64 × 84 nm |

Each step down roughly divides total voxels by 8 (2× in each of 3 dimensions).

---

## Why Voxel Size Is Not Uniform Across Axes (Anisotropy)

The Z axis has a slightly different physical size than X and Y (5.24 nm vs 4.0 nm at s0).
This is a property of how FIB-SEM works — the ion beam mills material away in Z slices,
and the slice thickness is not always perfectly matched to the X/Y pixel size.

This matters for coordinate math. If you know a location in nanometers (e.g. from
neuroglancer) and want to convert to voxel coordinates, you must divide each axis by its
own scale factor:

```
z_voxel = z_nm / 5.24    (at s0) or z_nm / 21    (at s2)
y_voxel = y_nm / 4.0     (at s0) or y_nm / 16    (at s2)
x_voxel = x_nm / 4.0     (at s0) or x_nm / 16    (at s2)
```

Using a single divisor for all three axes will give wrong results for Z.

---

## What Chunks Are

Zarr and N5 do not store the entire volume as one flat file. They break it into fixed-size
blocks called **chunks**, each stored as a separate object in S3.

For the jrc_hela-2 raw EM at s2, chunks are 64 × 64 × 64 voxels. For `mito_seg/s2`,
chunks are 96 × 96 × 96 voxels. Each chunk is stored and retrieved independently.

When you request a subvolume (a slice of the array), zarr:
1. Figures out which chunks overlap your requested region
2. Downloads exactly those chunks from S3
3. Assembles them and trims to your exact request

---

## Why Aligning to Chunk Boundaries Matters

If you request a 64-voxel-wide subvolume but the chunks are 96 voxels wide, zarr fetches
an entire 96-voxel chunk and discards the extra 32 voxels. You transferred 50% more data
than you used.

If you request a region that spans two chunks, zarr fetches both chunks even if only a
few voxels from the second chunk fall within your ROI. In the worst case, a request that
is just 1 voxel wider than a chunk boundary doubles the data transferred.

This is why the Sprint 1 ROI was updated from 64³ to 96³ — matching the chunk size of
`mito_seg/s2` means each patch request fetches exactly one chunk with no waste.

For training patch sampling, alignment to chunk boundaries is a performance optimization,
not a correctness requirement. Misaligned patches are still valid — they just cost more
network bandwidth.

---

## Why Some Numbers Seem to Get Larger with Downsampling

This is counterintuitive at first. A few cases where it comes up:

**Instance IDs in segmentation arrays:**
`mito_seg/s2` contains values like {0, 47, 110, 138}. These are not counts or
measurements — they are arbitrary integer labels assigned to individual mitochondria
(mitochondrion #47, #110, #138). The numbers are large because they were assigned
sequentially across the entire segmentation pipeline, which processed many objects before
reaching this region. A higher ID does not mean a bigger or more important mitochondrion.

**Chunk counts:**
At lower resolution, the volume has fewer voxels but the chunk size stays the same. This
means there are fewer total chunks. However, each individual chunk covers a larger physical
region of the cell, so a single chunk fetch at s4 gives you more of the cell than a single
chunk fetch at s0 — even though the data transfer is the same size in bytes.

**Foreground voxel counts across resolution levels:**
The number of foreground voxels goes down with downsampling (fewer total voxels), but the
*fraction* of foreground stays roughly similar — because the physical volume of mitochondria
relative to total cell volume does not change just because you're looking at a lower-resolution
representation of it. A mitochondrion that occupies 5% of the cell at s0 should still occupy
roughly 5% at s2.

If the foreground fraction changes dramatically between resolution levels (as `mito_pred` did
— 0.26% at s2 vs what should have been ~5%), that is a signal that something is wrong with
the label semantics at that level, not a normal consequence of downsampling.
