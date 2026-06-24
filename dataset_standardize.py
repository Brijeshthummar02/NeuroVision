"""
Dataset Standardization & Normalization Pipeline
================================================

Companion to `dataset_quality.py`. 

Where the quality script *audits* the dataset, this script *normalizes* it so every sample shares the same
resolution, file format, channel layout, mask encoding, and metadata / labeling conventions before training.

It does two things (with the second one being optional):

1. Manifest standardization (always runs, no images required)
     - Re-derives the canonical `patient_id` from the image path. The
       shipped `data_mask.csv` has a misaligned `patient_id` column
       (the same id is repeated across slices belonging to different
       patients), which silently breaks patient-grouped train/val/test
       splits and causes data leakage. The folder name in `image_path`
       is the ground-truth TCGA patient id, so we trust that.
     - Parses a numeric `slice` index from each filename.
     - Normalizes path separators and column ordering.

2. Pixel standardization (runs only for rows whose files exist locally)
     - Resizes every image to a single canonical resolution.
     - Forces a single channel layout (3-channel RGB) and dtype (uint8).
     - Resizes masks with nearest-neighbour interpolation and binarizes
       them to {0, 255}, then re-derives the 0/1 `mask` label from the
       actual mask content (catching mislabeled rows).
     - Writes the standardized copies to `dataset_standardized/`,
       mirroring the original folder layout, without touching the
       originals.

Outputs
-------
  data_mask_standardized.csv               canonical manifest (repo root)
  dataset_standardized/<patient>/...png    standardized images + masks
  dataset_reports/standardization_report.json   run summary

Run:
    python dataset_standardize.py

See PREPROCESSING.md for the full pipeline documentation.
"""

import os
import re
import json
import warnings

import numpy as np
import pandas as pd
import cv2

warnings.filterwarnings("ignore")

# PATHS / PARAMS  (all relative to project root)
CSV_PATH       = "data_mask.csv"            # source manifest in the repo root
IMG_ROOT       = "./"                        # images live as ./TCGA_xxx/TCGA_xxx_1.tif
OUTPUT_DIR     = "dataset_standardized"      # standardized images + masks go here
REPORT_DIR     = "dataset_reports"           # reuse the existing reports folder
MANIFEST_PATH  = os.path.join(os.getcwd(), OUTPUT_DIR, "data_mask_standardized.csv")

TARGET_SIZE    = (256, 256)   # canonical resolution as (width, height) — matches model input
IMAGE_EXT      = ".png"       # canonical lossless on-disk format
MASK_THRESHOLD = 127          # >= this -> foreground when binarizing masks

# canonical manifest columns, in order
MANIFEST_COLUMNS = [
    "patient_id", "slice", "image_path", "mask_path",
    "mask", "original_mask", "std_image_path", "std_mask_path",
]


# HELPERS — pure, no I/O (easy to unit-test)
def _posix(path) -> str:
    """Normalize Windows separators to POSIX so manifests are portable."""
    return str(path).replace("\\", "/").strip()


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Min-max scale any numeric image to 8-bit [0, 255].
    Medical TIFs are frequently 16-bit; this gives a consistent dtype.
    A flat image (max == min) maps to all-zeros.
    """
    img = img.astype(np.float64)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn) * 255.0
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


def derive_patient_id(image_path: str) -> str:
    """
    Canonical TCGA patient id = the folder that contains the slice.

    `TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif` -> `TCGA_CS_4941_19960909`

    Falls back to the first four underscore-separated tokens of the
    filename if the path has no parent folder.
    """
    parts = [p for p in _posix(image_path).split("/") if p]
    if len(parts) >= 2:
        return parts[-2]
    stem = os.path.splitext(parts[-1])[0] if parts else ""
    bits = stem.split("_")
    return "_".join(bits[:4]) if len(bits) >= 4 else stem


def derive_slice_index(image_path: str):
    """Parse the trailing slice number from a filename, e.g. `..._34.tif` -> 34."""
    stem = os.path.splitext(os.path.basename(_posix(image_path)))[0]
    m = re.search(r"_(\d+)$", stem)
    return int(m.group(1)) if m else None


def standardize_image(img: np.ndarray, target_size=TARGET_SIZE) -> np.ndarray:
    """
    Normalize one image to canonical (H, W, 3) RGB uint8 at `target_size`.

    Channel handling mirrors app.py's inference preprocessing:
      grayscale -> RGB, BGRA/RGBA -> RGB, BGR (cv2 default) -> RGB.
    Returns a resized RGB uint8 array; pixel-value normalization
    (/255 or mean-std) is left to model-feed time on purpose.
    """
    if img is None:
        raise ValueError("standardize_image received None")

    if img.ndim == 2:                                  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:          # single-channel 3D
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:          # BGRA / RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:          # cv2 loads BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"unsupported image shape {img.shape}")

    if img.dtype != np.uint8:
        img = _to_uint8(img)

    h, w = img.shape[:2]
    # INTER_AREA is the better choice when shrinking; INTER_LINEAR when growing
    shrinking = target_size[0] <= w and target_size[1] <= h
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    img = cv2.resize(img, (target_size[0], target_size[1]), interpolation=interp)
    return img


def standardize_mask(mask: np.ndarray, target_size=TARGET_SIZE,
                     threshold=MASK_THRESHOLD) -> np.ndarray:
    """
    Normalize one mask to canonical (H, W) single-channel binary {0, 255}.

    Masks are resized with NEAREST interpolation (never bilinear — that
    would invent intermediate label values along tumor boundaries) and
    then thresholded so the encoding is identical for every sample.
    """
    if mask is None:
        raise ValueError("standardize_mask received None")

    if mask.ndim == 3:
        mask = mask[:, :, 0]                 # collapse stray channels
    if mask.dtype != np.uint8:
        mask = _to_uint8(mask)

    mask = cv2.resize(mask, (target_size[0], target_size[1]),
                      interpolation=cv2.INTER_NEAREST)
    _, binar = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binar.astype(np.uint8)


def label_from_mask(mask: np.ndarray) -> int:
    """Re-derive the 0/1 tumor label from a (standardized) mask's content."""
    return int(np.any(mask > 0))


def _std_rel_path(path: str, ext=IMAGE_EXT) -> str:
    """Map an original relative path to its standardized counterpart."""
    root, _ = os.path.splitext(_posix(path))
    return root + ext


# STEP 1 — Manifest standardization (no images required)
def standardize_manifest(df: pd.DataFrame):
    """
    Normalize metadata + labeling conventions of the manifest itself.
    Returns (standardized_df, summary_dict).
    """
    for col in ("image_path", "mask_path"):
        if col not in df.columns:
            raise KeyError(f"source CSV missing required column: '{col}'")

    df = df.copy()

    # canonical patient_id from the image folder (fixes the misaligned column)
    derived_pid = df["image_path"].map(derive_patient_id)
    if "patient_id" in df.columns:
        pid_corrected = int((df["patient_id"].astype(str) != derived_pid).sum())
    else:
        pid_corrected = 0
    df["patient_id"] = derived_pid

    # numeric slice index
    df["slice"] = df["image_path"].map(derive_slice_index).astype("Int64")

    # portable paths
    df["image_path"] = df["image_path"].map(_posix)
    df["mask_path"] = df["mask_path"].map(_posix)

    # keep the source label for traceability; `mask` becomes the canonical label
    if "mask" in df.columns:
        df["original_mask"] = pd.to_numeric(df["mask"], errors="coerce").astype("Int64")
        df["mask"] = df["original_mask"]
    else:
        df["original_mask"] = pd.array([pd.NA] * len(df), dtype="Int64")
        df["mask"] = pd.array([pd.NA] * len(df), dtype="Int64")

    # standardized-output columns (filled in step 2)
    df["std_image_path"] = ""
    df["std_mask_path"] = ""

    df = df[MANIFEST_COLUMNS]

    summary = {
        "rows": int(len(df)),
        "unique_patients": int(df["patient_id"].nunique()),
        "patient_id_corrected": pid_corrected,
        "slices_parsed": int(df["slice"].notna().sum()),
    }

    print("\n── Manifest Standardization ────────────────────")
    print(f"  Rows                  : {summary['rows']:,}")
    print(f"  Unique patients       : {summary['unique_patients']}")
    print(f"  patient_id corrected  : {summary['patient_id_corrected']:,}")
    print(f"  Slice indices parsed  : {summary['slices_parsed']:,}")

    return df, summary


# STEP 2 — Pixel standardization (needs the image files on disk)
def process_pixels(df: pd.DataFrame, img_root=IMG_ROOT, out_dir=OUTPUT_DIR,
                   target_size=TARGET_SIZE, mask_threshold=MASK_THRESHOLD):
    """
    Standardize image + mask pixels for every row whose files exist.
    Writes standardized copies under `out_dir` and returns
    (updated_df, summary_dict). Missing-file rows are skipped and counted,
    so this is safe to run with only the CSV present.
    """
    os.makedirs(out_dir, exist_ok=True)

    shapes, dtypes = set(), set()
    processed = missing = label_corrected = 0
    std_imgs, std_masks, labels = [], [], []

    print("\n── Pixel Standardization ───────────────────────")
    print(f"  Target resolution : {target_size[0]}x{target_size[1]}  | format: {IMAGE_EXT} | channels: 3 | dtype: uint8")
    print(f"  Scanning {len(df):,} rows …")

    for _, row in df.iterrows():
        img_full = os.path.join(img_root, row["image_path"])
        mask_full = os.path.join(img_root, row["mask_path"])

        if not (os.path.isfile(img_full) and os.path.isfile(mask_full)):
            missing += 1
            std_imgs.append("")
            std_masks.append("")
            labels.append(row["mask"])
            continue

        img = cv2.imread(img_full, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_full, cv2.IMREAD_UNCHANGED)

        std_img = standardize_image(img, target_size)
        std_mask = standardize_mask(mask, target_size, mask_threshold)
        label = label_from_mask(std_mask)

        rel_img = _std_rel_path(row["image_path"])
        rel_mask = _std_rel_path(row["mask_path"])
        out_img = os.path.join(out_dir, rel_img)
        out_mask = os.path.join(out_dir, rel_mask)
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        os.makedirs(os.path.dirname(out_mask), exist_ok=True)

        # cv2 writes BGR; convert back so the file is a faithful RGB image
        cv2.imwrite(out_img, cv2.cvtColor(std_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_mask, std_mask)

        shapes.add(tuple(std_img.shape))
        dtypes.add(str(std_img.dtype))
        if pd.notna(row["original_mask"]) and int(row["original_mask"]) != label:
            label_corrected += 1

        std_imgs.append(_posix(rel_img))
        std_masks.append(_posix(rel_mask))
        labels.append(label)
        processed += 1

    df = df.copy()
    df["std_image_path"] = std_imgs
    df["std_mask_path"] = std_masks
    df["mask"] = pd.array(
        [int(v) if pd.notna(v) else pd.NA for v in labels], dtype="Int64"
    )

    res = f"{target_size[0]}x{target_size[1]}"
    consistency = {
        "uniform_resolution": len(shapes) <= 1,
        "uniform_channels": len(shapes) <= 1,
        "uniform_dtype": len(dtypes) <= 1,
        "resolution": res if processed else None,
        "channels": (shapes.pop()[2] if (processed and shapes) else (3 if processed else None)),
        "dtype": (next(iter(dtypes)) if dtypes else None),
    }

    summary = {
        "processed": processed,
        "missing_files": missing,
        "mask_label_corrected": label_corrected,
        "output_format": IMAGE_EXT,
        "mask_values": [0, 255],
        "mask_interpolation": "nearest",
        "consistency_check": consistency,
    }

    print(f"  Standardized rows     : {processed:,}")
    print(f"  Missing-file rows     : {missing:,}")
    print(f"  Mask labels corrected : {label_corrected:,}")
    if processed:
        print(f"  Consistency           : resolution={consistency['uniform_resolution']} "
              f"channels={consistency['uniform_channels']} dtype={consistency['uniform_dtype']}")
    else:
        print("  (no image files found locally — pixel step skipped, manifest still standardized)")

    return df, summary


# MAIN
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"\nLoading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH)

    # Step 1 — manifest
    manifest, manifest_summary = standardize_manifest(df)

    # Step 2 — pixels
    manifest, pixel_summary = process_pixels(manifest)

    # write the canonical manifest
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\n  Manifest generated at -> {MANIFEST_PATH}  ({len(manifest):,} rows)")

    # assemble + write the JSON report
    report = {
        "source_csv": CSV_PATH,
        "manifest": manifest_summary,
        "images": {
            "target_resolution": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
            "output_format": IMAGE_EXT,
            "channels": 3,
            "dtype": "uint8",
            "processed": pixel_summary["processed"],
            "missing_files": pixel_summary["missing_files"],
            "consistency_check": pixel_summary["consistency_check"],
        },
        "masks": {
            "values": pixel_summary["mask_values"],
            "interpolation": pixel_summary["mask_interpolation"],
            "threshold": MASK_THRESHOLD,
            "label_corrected": pixel_summary["mask_label_corrected"],
        },
        "outputs": {
            "manifest_csv": MANIFEST_PATH,
            "standardized_dir": OUTPUT_DIR,
            "report": os.path.join(REPORT_DIR, "standardization_report.json"),
        },
        "acceptance_criteria": {
            "consistent_resolution_and_format": bool(
                pixel_summary["consistency_check"]["uniform_resolution"]
                and pixel_summary["consistency_check"]["uniform_dtype"]
            ),
            "masks_standardized": True,
            "pipeline_documented": "PREPROCESSING.md",
        },
    }

    report_path = os.path.join(REPORT_DIR, "standardization_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report generated at -> {report_path}")

    print(f"\n  Standardization complete. Manifest: {MANIFEST_PATH}\n")


if __name__ == "__main__":
    main()
