# Sprint 0 — Skeleton

**Goal:** A script that runs end to end without errors, touching every major component with pass-through or stub functionality. The repo is immediately runnable by anyone who clones it. Sprint 0 also includes a data exploration step whose output will drive Sprint 1 planning — no Sprint 1 planning doc is finalized until `explore.py` has been run and its output reviewed.

**Exit conditions:**
- `python explore.py` runs without errors and produces a readable summary of the N5 group structure for jrc_hela-2
- `python train.py` runs without errors end to end using stubs

---

## Tasks

### Step 0 — Repo Setup
- [ ] Initialize git repo, create GitHub remote for `mitotrain`
- [ ] Add `README.md` and `PLANNING.md`
- [ ] Add `.gitignore` (Python standard + `checkpoints/`, `outputs/`)
- [ ] Add `requirements.txt` with initial dependencies:
  ```
  zarr
  s3fs
  fsspec[s3]
  dask
  numpy
  torch
  matplotlib
  fibsem_tools
  ```

### Step 1 — Directory Structure
Create the following empty directories and placeholder files:

```
mitotrain/
├── train.py
├── explore.py             # data exploration script — run before Sprint 1
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
├── PLANNING.md
├── SPRINT_0.md            # this document
└── README.md
```

### Step 2 — Data Exploration (`explore.py`)

Run `explore.py` to verify S3 access and map the actual N5 structure of jrc_hela-2 before writing any pipeline code. This script checks both S3 buckets, lists all array paths and shapes, and surfaces the ground truth crop names and sizes needed to plan Sprint 1.

**Install dependencies first:**
```bash
pip install "fsspec[s3]" zarr dask fibsem_tools
```

**Run:**
```bash
python explore.py
```

**Expected output includes:**
- Confirmation that both `janelia-cosem-publications` and `janelia-cosem-datasets` buckets are accessible anonymously
- Full N5 group tree for jrc_hela-2 in each bucket
- Shape, dtype, and chunk size for raw EM at multiple resolution levels (`s0`, `s1`, `s2`, `s4`)
- Ground truth crop names under `volumes/labels/0003/` and their array shapes
- Shape of mitochondria segmentation layer at multiple resolutions
- Confirmation that `fibsem_tools.read()` opens the data cleanly

**After running:** paste the full output into a new file `SPRINT_1_RESEARCH.md` in the repo root. This file is the basis for Sprint 1's planning doc. Do not finalize Sprint 1 until this step is complete.

**Key questions `explore.py` should answer:**
- Which bucket should be used for training labels — publications or datasets?
- What are the ground truth crop names and shapes for jrc_hela-2?
- What downsampled resolution level (`s1`, `s2`, `s4`) is practical for patch sampling on CPU?
- Does `fibsem_tools` open both buckets cleanly, or is raw `zarr.N5FSStore` more reliable?

### Step 3 — Stubs

Each module should contain a stub function that accepts the expected inputs and returns a dummy output of the correct type and shape. The goal is for `train.py` to call each one in sequence without errors — not for any of them to do real work yet.

**`data/loader.py`**
```python
def load_subvolume(dataset, layer, roi):
    """
    Stub: returns a random numpy array in place of a real subvolume.
    Shape: (1, 64, 64, 64) — one channel, 64^3 voxels.
    """
```

**`data/sampler.py`**
```python
def sample_patches(raw, labels, patch_size, n_patches):
    """
    Stub: returns a list of n_patches random (raw_patch, label_patch) tuples
    of shape (1, patch_size, patch_size, patch_size).
    """
```

**`model/unet.py`**
```python
def get_model():
    """
    Stub: returns a minimal nn.Module with a forward() that passes input
    through unchanged (identity model). Correct input/output shapes only.
    """
```

**`utils/visualize.py`**
```python
def save_slice(raw, labels, path):
    """
    Stub: creates a blank matplotlib figure and saves it to path.
    """
```

### Step 4 — `train.py` Entry Point

Wire the stubs together in the correct execution order. Every real training run will follow this same sequence — stubs will be replaced with real implementations sprint by sprint.

```
1. Load subvolume (raw EM + labels)          ← data/loader.py
2. Generate binary mask from labels          ← inline or data/loader.py
3. Sample patches                            ← data/sampler.py
4. Instantiate model                         ← model/unet.py
5. Forward pass on one batch                 ← inline
6. Compute loss                              ← inline
7. Save visualization                        ← utils/visualize.py
8. Print summary                             ← inline
```

Print a short status line at each step so progress is visible when the script runs:

```
[1/7] Loading subvolume...        shape: (1, 64, 64, 64)
[2/7] Generating binary mask...   mito voxels: 0 (stub)
[3/7] Sampling patches...         n=8, shape: (1, 32, 32, 32)
[4/7] Instantiating model...      params: 0 (stub)
[5/7] Forward pass...             output shape: (1, 1, 32, 32, 32)
[6/7] Computing loss...           loss: 0.0000 (stub)
[7/7] Saving visualization...     outputs/slice_preview.png
Done.
```

---

## Commit Plan

| Commit | Message |
|--------|---------|
| 1 | `init: repo structure, README, PLANNING` |
| 2 | `sprint0: add explore.py and requirements` |
| 3 | `sprint0: add stub modules and train.py skeleton` |
| 4 | `sprint0: explore.py runs, output saved to SPRINT_1_RESEARCH.md` |
| 5 | `sprint0: train.py runs end to end with stubs` |

---

## Notes

- Shapes in `train.py` stubs are hardcoded constants for now — they will be updated after `explore.py` confirms realistic sizes, and will become config values in Sprint 4.
- The stub for `unet.py` is temporary. The funkelab U-Net replaces it in Sprint 3.
- `explore.py` is a development/research tool, not part of the training pipeline. It does not need to be kept in sync with `train.py`.
- No model weights are saved in this sprint.
- `SPRINT_1_RESEARCH.md` is gitignored or committed as raw notes — it is not a polished document.
