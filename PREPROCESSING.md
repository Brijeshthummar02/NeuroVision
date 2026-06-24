# Dataset Standardization & Preprocessing Pipeline

This document describes how raw MRI data is standardized into a single,
consistent format before training — covering image dimensions, file
formats, channel layout, mask encoding, and metadata / labeling
conventions.

It is implemented by [`dataset_standardize.py`](dataset_standardize.py)
and is the normalization counterpart to the auditing performed by
[`dataset_quality.py`](dataset_quality.py).

> Closes #70 — *Standardize and Normalize Dataset Formats.*

---

## Why this exists

The raw TCGA data and the shipped `data_mask.csv` are not uniform:

- **Mixed image properties.** Source scans/masks are TIFs that may be
  grayscale, RGB, RGBA, 8-bit or 16-bit, and not guaranteed to all be the
  exact same size.
- **Inconsistent mask encoding.** Masks can carry stray channels or
  non-binary values, and the only thing that matters for training is a
  clean binary tumor region.
- **Broken `patient_id` metadata.** In the shipped `data_mask.csv` the
  `patient_id` column is misaligned — the same id is repeated across
  slices that actually belong to *different* patients (3,894 of 3,929
  rows). A patient-grouped train/val/test split keyed on that column
  would leak the same patient across splits and inflate metrics.

The pipeline removes all of these inconsistencies non-destructively: the
originals are never modified; standardized copies and a corrected
manifest are written alongside them.

---

## What "standardized" means here

| Property        | Canonical value                              |
|-----------------|----------------------------------------------|
| Image size      | **256 × 256** (matches the model input)      |
| Image channels  | **3 (RGB)**                                  |
| Image dtype     | **uint8** (16-bit inputs min-max scaled)     |
| Image format    | **PNG** (lossless)                           |
| Mask size       | **256 × 256**, resized **nearest-neighbour** |
| Mask channels   | **1 (grayscale)**                            |
| Mask values     | **{0, 255}** (binary)                        |
| `patient_id`    | Derived from the image folder (ground truth) |
| `slice`         | Parsed integer slice index                   |
| `mask` label    | Re-derived from actual mask content (0/1)    |

Pixel-value normalization (`/255` for classification, mean–std
standardization for segmentation) is intentionally **not** baked into the
stored files — it stays at model-feed time, exactly as in
`app.py::preprocess_image_classification` / `preprocess_image_segmentation`,
so the standardized data stays reusable across both stages.

---

## Pipeline stages

### Stage 1 — Manifest standardization (no images required)

Runs purely on the CSV, so it works even in a fresh clone without the
dataset downloaded:

1. Re-derive the canonical `patient_id` from each `image_path` folder.
2. Parse a numeric `slice` index from each filename (`..._34.tif → 34`).
3. Normalize path separators (Windows `\` → POSIX `/`).
4. Preserve the source label as `original_mask` for traceability and
   promote `mask` to the canonical label column.
5. Emit a fixed, ordered set of columns.

### Stage 2 — Pixel standardization (requires the image files)

For every row whose files exist locally:

1. **Image** → 3-channel RGB, uint8, resized to 256 × 256
   (`INTER_AREA` when shrinking, `INTER_LINEAR` when enlarging).
2. **Mask** → single-channel, resized to 256 × 256 with **`INTER_NEAREST`**
   (bilinear would invent gray values along tumor boundaries), then
   thresholded to a clean binary `{0, 255}`.
3. **Label** → re-derived from the standardized mask (`1` if any
   foreground pixel, else `0`), flagging rows whose original label was
   wrong.
4. Standardized copies are written to `dataset_standardized/`, mirroring
   the original `TCGA_*/` folder layout, as `.png`.

Rows whose files are missing locally are skipped and counted — the run
still succeeds and the manifest is still corrected.

---

## Usage

```bash
python dataset_standardize.py
```

No arguments; behaviour is controlled by the constants at the top of the
script (`CSV_PATH`, `IMG_ROOT`, `OUTPUT_DIR`, `TARGET_SIZE`, `IMAGE_EXT`,
`MASK_THRESHOLD`).

### Outputs

| Path                                          | Contents                                  |
|-----------------------------------------------|-------------------------------------------|
| `data_mask_standardized.csv`                  | Canonical manifest (see schema below)     |
| `dataset_standardized/<patient>/*.png`        | Standardized images + masks (gitignored)  |
| `dataset_reports/standardization_report.json` | Run summary + consistency check           |

> `dataset_standardized/` is git-ignored — standardized images are
> regenerable and, like the raw `TCGA_*` folders, are never committed.

### Manifest schema (`data_mask_standardized.csv`)

| Column            | Description                                            |
|-------------------|--------------------------------------------------------|
| `patient_id`      | Canonical TCGA patient id (corrected)                  |
| `slice`           | Integer slice index parsed from the filename           |
| `image_path`      | Original image path (unchanged)                         |
| `mask_path`       | Original mask path (unchanged)                          |
| `mask`            | Canonical 0/1 label (re-derived from mask when present)|
| `original_mask`   | Label as it appeared in the source CSV                  |
| `std_image_path`  | Standardized image path under `dataset_standardized/`  |
| `std_mask_path`   | Standardized mask path under `dataset_standardized/`   |

The original `image_path` / `mask_path` columns are preserved so existing
code keeps working; training can switch to the `std_*` columns once the
standardized set has been generated.

---

## Using it in training

```python
import pandas as pd

df = pd.read_csv("data_mask_standardized.csv")

# Patient-grouped split — now safe, because patient_id is correct
from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(test_size=0.30, n_splits=1, random_state=42)
train_idx, hold_idx = next(splitter.split(df, groups=df["patient_id"]))
```

---

## Sample report

```json
{
  "manifest": {
    "rows": 3929,
    "unique_patients": 110,
    "patient_id_corrected": 3894,
    "slices_parsed": 3929
  },
  "images": {
    "target_resolution": "256x256",
    "output_format": ".png",
    "channels": 3,
    "dtype": "uint8",
    "consistency_check": { "uniform_resolution": true, "uniform_dtype": true }
  },
  "masks": { "values": [0, 255], "interpolation": "nearest", "threshold": 127 },
  "acceptance_criteria": {
    "consistent_resolution_and_format": true,
    "masks_standardized": true,
    "pipeline_documented": "PREPROCESSING.md"
  }
}
```

---

## Tests

Unit tests live in
[`tests/unit/test_dataset_standardize.py`](tests/unit/test_dataset_standardize.py):

```bash
pytest tests/unit/test_dataset_standardize.py
```

They cover the metadata helpers, image/mask standardization (including the
nearest-neighbour guarantee and uint16 → uint8 scaling), manifest
correction, and an end-to-end pixel run on synthetic patient folders.
