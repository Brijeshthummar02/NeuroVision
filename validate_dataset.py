

import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

log = logging.getLogger("neurovision.validate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

EXPECTED_SIZE      = 256
EXPECTED_CHANNELS  = 3
EXPECTED_IMG_MODE  = "RGB"
EXPECTED_MASK_MODE = "L"
EXPECTED_FORMAT    = "PNG"
MASK_POSITIVE_VAL  = 255
VALID_MASK_VALUES  = {0, 255}
VALID_LABELS       = {0, 1}


class CheckResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def ok(self, msg):
        self.passed.append(msg)
        log.info("  ✓  %s", msg)

    def fail(self, msg):
        self.failed.append(msg)
        log.error("  ✗  %s", msg)

    def warn(self, msg):
        self.warnings.append(msg)
        log.warning("  ⚠  %s", msg)

    @property
    def success(self):
        return len(self.failed) == 0

def check_metadata_exists(processed_dir: Path, r: CheckResult) -> pd.DataFrame | None:
    meta_path = processed_dir / "metadata.csv"
    if not meta_path.exists():
        r.fail("metadata.csv not found")
        return None
    df = pd.read_csv(meta_path)
    r.ok(f"metadata.csv found ({len(df)} rows)")
    return df


def check_metadata_schema(df: pd.DataFrame, r: CheckResult):
    required_cols = {
        "patient_id", "image_path", "mask_path", "label", "has_mask",
        "img_width", "img_height", "img_channels", "img_format",
        "mask_format", "tumour_pixels", "tumour_fraction",
    }
    missing = required_cols - set(df.columns)
    if missing:
        r.fail(f"metadata.csv missing columns: {missing}")
    else:
        r.ok("metadata.csv has all required columns")

    bad_labels = df[~df["label"].isin(VALID_LABELS)]
    if len(bad_labels):
        r.fail(f"{len(bad_labels)} rows have invalid label values (expected 0 or 1)")
    else:
        r.ok("All label values are valid {0, 1}")

    has_mask_mismatch = df[
        ((df["has_mask"] == 1) & (df["tumour_pixels"] == 0)) |
        ((df["has_mask"] == 0) & (df["tumour_pixels"] > 0))
    ]
    if len(has_mask_mismatch):
        r.fail(f"{len(has_mask_mismatch)} rows have inconsistent has_mask vs tumour_pixels")
    else:
        r.ok("has_mask and tumour_pixels are consistent across all rows")


def check_image_files(processed_dir: Path, df: pd.DataFrame, r: CheckResult):
    bad_res = bad_mode = bad_fmt = missing = 0
    for row in tqdm(df.itertuples(), total=len(df), desc="Checking images", leave=False):
        img_path = processed_dir / row.image_path
        if not img_path.exists():
            missing += 1
            continue
        try:
            with Image.open(img_path) as im:
                if im.size != (EXPECTED_SIZE, EXPECTED_SIZE):
                    bad_res += 1
                if im.mode != EXPECTED_IMG_MODE:
                    bad_mode += 1
                if im.format and im.format != EXPECTED_FORMAT:
                    bad_fmt += 1
        except Exception:
            missing += 1

    if missing:
        r.fail(f"{missing} image files missing or unreadable")
    else:
        r.ok("All image files present and readable")

    if bad_res:
        r.fail(f"{bad_res} images have wrong resolution (expected {EXPECTED_SIZE}×{EXPECTED_SIZE})")
    else:
        r.ok(f"All images are {EXPECTED_SIZE}×{EXPECTED_SIZE}")

    if bad_mode:
        r.fail(f"{bad_mode} images not in RGB mode")
    else:
        r.ok("All images are RGB (3-channel)")

    if bad_fmt:
        r.warn(f"{bad_fmt} images report a non-PNG format header (may still be valid)")


def check_mask_files(processed_dir: Path, df: pd.DataFrame, r: CheckResult):
    mask_rows = df[df["has_mask"] == 1]
    if len(mask_rows) == 0:
        r.warn("No tumour masks found — dataset may be all-negative")
        return

    bad_binary = bad_size = bad_mode = missing = 0
    for row in tqdm(mask_rows.itertuples(), total=len(mask_rows), desc="Checking masks", leave=False):
        if not row.mask_path:
            missing += 1
            continue
        mask_path = processed_dir / row.mask_path
        if not mask_path.exists():
            missing += 1
            continue
        try:
            with Image.open(mask_path) as im:
                if im.size != (EXPECTED_SIZE, EXPECTED_SIZE):
                    bad_size += 1
                if im.mode != EXPECTED_MASK_MODE:
                    bad_mode += 1
                arr = np.array(im)
                unique_vals = set(arr.flatten().tolist())
                if not unique_vals.issubset(VALID_MASK_VALUES):
                    bad_binary += 1
        except Exception:
            missing += 1

    if missing:
        r.fail(f"{missing} mask files missing or unreadable")
    else:
        r.ok(f"All {len(mask_rows)} mask files present and readable")

    if bad_size:
        r.fail(f"{bad_size} masks have wrong resolution")
    else:
        r.ok(f"All masks are {EXPECTED_SIZE}×{EXPECTED_SIZE}")

    if bad_mode:
        r.fail(f"{bad_mode} masks not in single-channel (L) mode")
    else:
        r.ok("All masks are single-channel (L mode)")

    if bad_binary:
        r.fail(f"{bad_binary} masks contain values outside {{0, 255}} — not fully binarised")
    else:
        r.ok("All masks are binary {0, 255}")


def check_spatial_alignment(processed_dir: Path, df: pd.DataFrame, r: CheckResult):
    
    mask_rows = df[df["has_mask"] == 1].head(50)  # sample 50 pairs
    mismatches = 0
    for row in mask_rows.itertuples():
        img_path  = processed_dir / row.image_path
        mask_path = processed_dir / row.mask_path
        if not img_path.exists() or not mask_path.exists():
            continue
        with Image.open(img_path) as im, Image.open(mask_path) as mk:
            if im.size != mk.size:
                mismatches += 1

    if mismatches:
        r.fail(f"{mismatches}/50 sampled pairs have mismatched image↔mask sizes")
    else:
        r.ok("Spot-check: image and mask sizes are aligned")


def check_class_balance(df: pd.DataFrame, r: CheckResult):
    n_tumor    = int(df["has_mask"].sum())
    n_no_tumor = len(df) - n_tumor
    ratio = n_tumor / max(len(df), 1)
    msg = f"Class balance — tumour: {n_tumor} ({ratio:.1%}), no-tumour: {n_no_tumor}"
    if 0.3 <= ratio <= 0.7:
        r.ok(msg + " [balanced]")
    elif 0.15 <= ratio < 0.3 or 0.7 < ratio <= 0.85:
        r.warn(msg + " [mild imbalance — monitor training]")
    else:
        r.warn(msg + " [severe imbalance — consider oversampling or weighted loss]")


def check_no_duplicates(df: pd.DataFrame, r: CheckResult):
    if "source_image_md5" not in df.columns:
        r.warn("source_image_md5 column absent — cannot check for duplicates")
        return
    dupes = df[df.duplicated(subset="source_image_md5", keep=False)]
    if len(dupes):
        r.warn(f"{len(dupes)} rows share an MD5 hash — possible duplicate source files")
    else:
        r.ok("No duplicate source files detected (MD5 check)")




def run_validation(processed_dir: Path, strict: bool) -> bool:
    r = CheckResult()
    log.info("── Validating: %s ──", processed_dir)

    df = check_metadata_exists(processed_dir, r)
    if df is None:
        log.error("Cannot continue without metadata.csv")
        return False

    check_metadata_schema(df, r)
    check_image_files(processed_dir, df, r)
    check_mask_files(processed_dir, df, r)
    check_spatial_alignment(processed_dir, df, r)
    check_class_balance(df, r)
    check_no_duplicates(df, r)

    log.info("Results")
    log.info("  Passed   : %d", len(r.passed))
    log.info("  Warnings : %d", len(r.warnings))
    log.info("  Failed   : %d", len(r.failed))

    if strict and r.warnings:
        return False
    return r.success


def parse_args():
    p = argparse.ArgumentParser(description="NeuroVision dataset validator")
    p.add_argument("--processed_dir", required=True, help="Directory produced by preprocess.py")
    p.add_argument("--strict", action="store_true",
                   help="Treat warnings as failures")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = run_validation(Path(args.processed_dir), args.strict)
    sys.exit(0 if ok else 1)
