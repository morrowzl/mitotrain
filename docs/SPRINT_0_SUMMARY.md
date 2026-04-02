# Sprint 0 Summary — Skeleton

**Status:** Complete  
**Exit condition met:** `python train.py` runs end to end without errors.

---

## What Was Built

### Repo setup
- Added `checkpoints/`, `outputs/`, and `SPRINT_1_RESEARCH.md` to `.gitignore`
- Created `requirements.txt` with all Sprint 0–2 dependencies; funkelab U-Net is commented out with install instructions (not on PyPI — requires `git+https://github.com/funkelab/funlib.learn.torch.git`)

### Directory structure
```
mitotrain/
├── train.py
├── explore.py
├── data/
│   ├── __init__.py
│   ├── loader.py
│   └── sampler.py
├── model/
│   ├── __init__.py
│   └── unet.py
├── utils/
│   ├── __init__.py
│   └── visualize.py
├── checkpoints/           # gitignored
├── outputs/               # gitignored
└── docs/
    ├── PLANNING.md
    ├── SPRINT_0.md
    └── SPRINT_0_SUMMARY.md   # this file
```

### Stub modules

| Module | Stub behavior |
|--------|--------------|
| `data/loader.py` | Returns a random `(1, 64, 64, 64)` uint8 array |
| `data/sampler.py` | Returns `n_patches` random `(1, patch_size, patch_size, patch_size)` pairs |
| `model/unet.py` | Identity `nn.Module` — passes input through unchanged |
| `utils/visualize.py` | Saves a blank two-panel matplotlib figure to `outputs/` |

### train.py

Wires all stubs in order with step-by-step console output:

```
[1/7] Loading subvolume...        shape: (1, 64, 64, 64)
[2/7] Generating binary mask...   mito voxels: N (stub)
[3/7] Sampling patches...         n=8, shape: (1, 32, 32, 32)
[4/7] Instantiating model...      params: 0 (stub)
[5/7] Forward pass...             output shape: (1, 1, 32, 32, 32)
[6/7] Computing loss...           loss: X.XXXX (stub)
[7/7] Saving visualization...     outputs/slice_preview.png
Done.
```

---

## Decisions and Notes

- **funkelab U-Net** — `model/unet.py` documents the full Sprint 3 instantiation (channels, fmap counts, downsample factors) in its module docstring so the Sprint 3 implementation is ready to drop in. The stub itself is a zero-parameter identity module.
- **Valid convolutions** — the funkelab U-Net crops its output spatially due to valid (not padded) convolutions. A **132³ input patch** is a safe starting size with the suggested config; this should be verified with a shape test before committing to a patch size in Sprint 2/3.
- **Patch size** — hardcoded to 32³ for Sprint 0. Will be updated after `explore.py` confirms practical resolution levels, and will become a config constant in Sprint 4.

---

## Next Step

Before writing any Sprint 1 code, run `explore.py` and save the output:

```bash
pip install "fsspec[s3]" zarr dask fibsem_tools
python explore.py > docs/SPRINT_1_RESEARCH.md
```

This output drives all Sprint 1 decisions (which bucket, which resolution level, crop names and shapes).
