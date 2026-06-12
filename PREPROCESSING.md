# NeuroVision — Preprocessing Pipeline Documentation

> **Issue:** Create preprocessing workflows to standardize image dimensions,
> file formats, masks, metadata, and labelling conventions across all datasets.

---

## Table of Contents

1. [Overview](#overview)
2. [Standards Reference](#standards-reference)
3. [Pipeline Scripts](#pipeline-scripts)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [Image Standardisation](#image-standardisation)
6. [Mask Standardisation](#mask-standardisation)
7. [Metadata Schema](#metadata-schema)
8. [Labelling Conventions](#labelling-conventions)
9. [Validation Checks](#validation-checks)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Raw MRI data comes from the TCGA (The Cancer Genome Atlas) dataset.
Scans arrive as 16-bit or 8-bit TIFF files organised in per-patient directories.
Before any model training the data must be normalised to a single, guaranteed
specification so that the `DataGenerator` receives consistent input.

```
Raw TCGA data                   Preprocessing pipeline             Processed data
────────────────                ──────────────────────             ──────────────
TCGA_HT_.../                                                       TCGA_HT_.../
  image.tif      ─── load ──►  resize / convert / binarise ──►      image.png
  image_mask.tif ─── load ──►  binarise / resize            ──►      image_mask.png
                                                                   metadata.csv
```

---

## Standards Reference

| Property | Standard value |
|---|---|
| Image format | PNG (lossless) |
| Bit depth | 8-bit per channel |
| Image channels | 3 (RGB) |
| Spatial resolution | 256 × 256 pixels |
| Mask format | PNG (lossless) |
| Mask channels | 1 (grayscale / `L` mode) |
| Mask dtype | uint8 |
| Mask pixel values | **0** = background, **255** = tumour |
| Tumour label | `1` |
| No-tumour label | `0` |
| Resize interpolation (images) | `INTER_AREA` (anti-aliased) |
| Resize interpolation (masks) | `INTER_NEAREST` (no label blending) |

---

## Pipeline Scripts

| Script | Purpose |
|---|---|
| `preprocess.py` | Main pipeline — reads raw data, writes standardised output + `metadata.csv` |
| `validate_dataset.py` | Verifies a processed directory against every acceptance criterion |

### Dependencies

```
pip install numpy pandas pillow scikit-image opencv-python-headless tqdm albumentations scipy
```

---

## Step-by-Step Workflow

### 1  Run the preprocessor

```bash
python preprocess.py \
    --input_dir  /data/raw/TCGA          \
    --output_dir /data/processed/TCGA    \
    --img_size   256
```

Expected console output:

```
Discovered 3929 samples in /data/raw/TCGA
Preprocessing: 100%|████████| 3929/3929 [02:14<00:00, 29.2 sample/s]
Metadata written to /data/processed/TCGA/metadata.csv
── Summary ──────────────────────────────────
  Total samples  : 3929
  Tumour         : 1373 (34.9%)
  No tumour      : 2556 (65.1%)
  Errors         : 0
  Output dir     : /data/processed/TCGA
```

### 2  Validate the output

```bash
python validate_dataset.py --processed_dir /data/processed/TCGA
```

All checks must exit 0 before training.

### 3  (Optional) Dry-run validation on raw sources

```bash
python preprocess.py \
    --input_dir /data/raw/TCGA \
    --output_dir /tmp/unused   \
    --validate_only
```

---

## Image Standardisation

### Input handling

| Source condition | Handling |
|---|---|
| 16-bit TIFF | Linearly rescaled → float32 `[0, 255]` before uint8 cast |
| Grayscale (2-D) | Replicated across 3 channels |
| RGBA (4 channels) | Alpha channel dropped |
| Single-channel (H, W, 1) | Concatenated to (H, W, 3) |

### Resize strategy

Downscaling from typical clinical resolutions (512×512, 240×240) uses
`cv2.INTER_AREA` (pixel-area relation), which minimises aliasing artefacts
compared with bilinear or bicubic.

### Pixel value range

The preprocessor writes **uint8 PNG** files with values in `[0, 255]`.  
The `DataGenerator` then **zero-mean / unit-variance** normalises *at load time*
(per-image statistics, not dataset-wide).  This separation keeps the raw
processed files universally readable (in any image viewer) while still feeding
the network standardised tensors.

---

## Mask Standardisation

Masks follow a strict binary convention:

```
0   → background  (no tumour)
255 → foreground  (tumour tissue)
```

### Binarisation logic

```python
if mask.max() > 1.0:
    mask = mask / 255.0          # normalise to [0, 1]
binary = (mask > 0.5).astype(np.uint8) * 255
```

The 0.5 threshold works correctly for:
- Already-binary masks (values exactly 0 or 255)
- Probability maps (values in [0, 1])
- Masks with annotation noise (slight off-white = still 255)

### Resize strategy

Masks are resized with `cv2.INTER_NEAREST`.  This guarantees that no
intermediate pixel values (e.g. 127) are introduced by interpolation — the
output is strictly `{0, 255}`.

### Spatial alignment guarantee

Both image and mask are always resized to the same `--img_size` dimension,
so `image[i, j]` and `mask[i, j]` refer to the same anatomical location.

---

## Metadata Schema

`metadata.csv` — one row per processed sample.

| Column | Type | Description |
|---|---|---|
| `patient_id` | str | TCGA patient directory name |
| `image_path` | str | Path to processed image (relative to `output_dir`) |
| `mask_path` | str | Path to processed mask; empty string if no mask |
| `label` | int | `1` = tumour present, `0` = no tumour |
| `has_mask` | int | `1` if a mask file was produced, else `0` |
| `img_width` | int | Always `IMG_SIZE` (256) |
| `img_height` | int | Always `IMG_SIZE` (256) |
| `img_channels` | int | Always `3` |
| `img_format` | str | Always `PNG` |
| `mask_format` | str | `PNG` or `N/A` |
| `mask_dtype` | str | `uint8` |
| `mask_positive_val` | int | `255` |
| `tumour_pixels` | int | Count of pixels == 255 in mask |
| `background_pixels` | int | Count of pixels == 0 in mask |
| `tumour_fraction` | float | `tumour_pixels / (img_width * img_height)` |
| `source_image_md5` | str | MD5 hash of the raw source file |
| `processed_at` | str | UTC timestamp of preprocessing |

**Using metadata.csv in training:**

```python
import pandas as pd
df = pd.read_csv("processed/TCGA/metadata.csv")

# Filter to only samples with a mask
df_tumor = df[df["has_mask"] == 1]

# Train/val/test split (deterministic)
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.30, random_state=42,
                                     stratify=df["label"])
val_df, test_df   = train_test_split(test_df, test_size=0.50, random_state=42,
                                     stratify=test_df["label"])
```

---

## Labelling Conventions

| Convention | Value | Where used |
|---|---|---|
| Tumour present | `label = 1` | `metadata.csv`, DataGenerator, classifier head |
| No tumour | `label = 0` | `metadata.csv`, DataGenerator, classifier head |
| Mask foreground pixel | `255` (uint8) | PNG mask files |
| Mask background pixel | `0` (uint8) | PNG mask files |
| In-network binarisation | `y = (y > 0).astype(int)` | DataGenerator `__data_generation` |

The in-network binarisation step converts the `{0, 255}` mask loaded from
disk to the `{0, 1}` range that the sigmoid output and Dice/Tversky losses
expect.

---

## Validation Checks

`validate_dataset.py` runs the following checks automatically:

| Check | Criterion | Pass condition |
|---|---|---|
| metadata.csv exists | File present | ✓ |
| Schema completeness | All required columns present | ✓ |
| Label integrity | All labels in `{0, 1}` | ✓ |
| has_mask consistency | `has_mask` agrees with `tumour_pixels` | ✓ |
| Image presence | All `image_path` files exist | ✓ |
| Image resolution | Every image is 256×256 | ✓ |
| Image mode | Every image is RGB | ✓ |
| Mask presence | All `mask_path` files exist for `has_mask=1` rows | ✓ |
| Mask resolution | Every mask is 256×256 | ✓ |
| Mask mode | Every mask is single-channel (L) | ✓ |
| Mask binary values | Every mask pixel is in `{0, 255}` | ✓ |
| Spatial alignment | Image and mask sizes match (50-pair spot-check) | ✓ |
| Class balance | Ratio of tumour:no-tumour reported with advisory | ⚠ |
| Duplicate detection | MD5 hash check for identical source files | ⚠ |

Legend: ✓ = hard failure if violated  ⚠ = warning only

---

## Troubleshooting

**`OSError: cannot identify image file`**  
The TIFF may be corrupted or use a non-standard compression.  
→ Re-download the affected scan.  Run `python preprocess.py --validate_only`
to isolate all bad files before re-running the full pipeline.

**`AssertionError` in validate_dataset — mask contains value 127**  
The raw mask was saved with soft edges (anti-aliased annotation tool).  
→ The binarise_mask function uses threshold=0.5 which handles this; if the
problem persists, check the annotation source and re-export with hard edges.

**`bad_res` failures for some samples**  
If raw images are smaller than 256×256, `INTER_AREA` still works but quality
may be lower. Consider using `INTER_LANCZOS4` for upscaling instead.

**Severe class imbalance warning**  
Use the `class_weight` parameter in `model.fit()` or oversample the minority
class in the DataGenerator.

---

*Last updated: 2026-06-09 | Pipeline version: 1.0.0*
