
import os
import sys
import argparse
import logging
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageOps
from skimage import io, exposure
from tqdm import tqdm

DEFAULT_IMG_SIZE   = 256          
OUTPUT_IMAGE_FMT   = "PNG"        
MASK_POSITIVE_VAL  = 255          
MASK_NEGATIVE_VAL  = 0            
LABEL_TUMOR        = 1            
LABEL_NO_TUMOR     = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("neurovision.preprocess")


def load_image(path: Path) -> np.ndarray:
    img = io.imread(str(path))

    if img.dtype == np.uint16:
        img = (img / 65535.0 * 255).astype(np.float32)
    else:
        img = img.astype(np.float32)

    #3-channel
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:   # RGBA
        img = img[..., :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)

    return img  

def resize_image(img: np.ndarray, size: int) -> np.ndarray:
   
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalise_to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def standardise_image(path: Path, size: int) -> np.ndarray:
    img = load_image(path)
    img = resize_image(img, size)
    img = normalise_to_uint8(img)
    return img   # uint8, (size, size, 3)

def load_mask(path: Path) -> np.ndarray:

    mask = io.imread(str(path))
    if mask.dtype == np.uint16:
        mask = (mask / 65535.0 * 255).astype(np.float32)
    else:
        mask = mask.astype(np.float32)

    # Collapse channel dim
    if mask.ndim == 3:
        mask = mask.mean(axis=-1)

    return mask   # float32, H×W


def binarise_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:

    if mask.max() > 1.0:
        mask = mask / 255.0
    binary = (mask > threshold).astype(np.uint8) * MASK_POSITIVE_VAL
    return binary   # uint8, values in {0, 255}


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:

    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def standardise_mask(path: Path, size: int) -> np.ndarray:
    mask = load_mask(path)
    mask = binarise_mask(mask)
    mask = resize_mask(mask, size)
    return mask  


def has_tumour(mask: np.ndarray) -> bool:
   
    return int(mask.max()) == MASK_POSITIVE_VAL


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_mask_stats(mask: np.ndarray) -> dict:
   
    total = mask.size
    tumour_px = int((mask == MASK_POSITIVE_VAL).sum())
    return {
        "tumour_pixels":      tumour_px,
        "background_pixels":  total - tumour_px,
        "tumour_fraction":    round(tumour_px / total, 6),
    }
def discover_samples(input_dir: Path):

    samples = []
    for patient_dir in sorted(input_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name
        tifs = sorted(patient_dir.glob("*.tif"))
        masks   = [t for t in tifs if "_mask" in t.stem]
        images  = [t for t in tifs if "_mask" not in t.stem]

        if not images:
            log.warning("No image found in %s — skipping", patient_dir)
            continue

        for img_path in images:
           
            base = img_path.stem
            mask_path = patient_dir / f"{base}_mask.tif"
            if not mask_path.exists():
               
                mask_path = masks[0] if masks else None

            samples.append((pid, img_path, mask_path))

    return samples


def process_sample(
    pid: str,
    img_path: Path,
    mask_path: Path | None,
    out_dir: Path,
    size: int,
) -> dict:

    patient_out = out_dir / pid
    patient_out.mkdir(parents=True, exist_ok=True)

    img_arr = standardise_image(img_path, size)
    out_img_name = img_path.stem + ".png"
    out_img_path = patient_out / out_img_name
    Image.fromarray(img_arr).save(out_img_path, format=OUTPUT_IMAGE_FMT)

    
    if mask_path and mask_path.exists():
        mask_arr = standardise_mask(mask_path, size)
        tumour    = has_tumour(mask_arr)
        mask_stats = compute_mask_stats(mask_arr)
        out_mask_name = mask_path.stem + ".png"
        out_mask_path = patient_out / out_mask_name
        Image.fromarray(mask_arr).save(out_mask_path, format=OUTPUT_IMAGE_FMT)
        
        mask_rel = str(out_mask_path.relative_to(out_dir))
    else:
        tumour     = False
        mask_stats = {"tumour_pixels": 0, "background_pixels": size * size, "tumour_fraction": 0.0}
        mask_rel   = ""

    label = LABEL_TUMOR if tumour else LABEL_NO_TUMOR

    return {
        "patient_id":        pid,
        "image_path":        str(out_img_path.relative_to(out_dir)),
        "mask_path":         mask_rel,
        "label":             label,
        "has_mask":          int(tumour),
        "img_width":         size,
        "img_height":        size,
        "img_channels":      3,
        "img_format":        "PNG",
        "mask_format":       "PNG" if mask_rel else "N/A",
        "mask_dtype":        "uint8",
        "mask_positive_val": MASK_POSITIVE_VAL,
        **mask_stats,
        "source_image_md5":  file_md5(img_path),
        "processed_at":      datetime.utcnow().isoformat(timespec="seconds"),
    }


def run_pipeline(input_dir: Path, output_dir: Path, size: int, validate_only: bool):
    """Entry point — discover, process, and write metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(input_dir)
    log.info("Discovered %d samples in %s", len(samples), input_dir)

    if validate_only:
        log.info("--validate_only set: checking source files, not writing output.")
        _validate_sources(samples)
        return

    records = []
    errors  = []

    for pid, img_path, mask_path in tqdm(samples, desc="Preprocessing", unit="sample"):
        try:
            rec = process_sample(pid, img_path, mask_path, output_dir, size)
            records.append(rec)
        except Exception as exc:
            log.error("FAILED %s/%s: %s", pid, img_path.name, exc)
            errors.append({"patient_id": pid, "file": str(img_path), "error": str(exc)})

    # Write metadata CSV
    df = pd.DataFrame(records)
    meta_path = output_dir / "metadata.csv"
    df.to_csv(meta_path, index=False)
    log.info("Metadata written to %s", meta_path)

    # Summary stats
    n_tumor    = int(df["has_mask"].sum())
    n_no_tumor = len(df) - n_tumor
    log.info("── Summary ")
    log.info("  Total samples  : %d", len(df))
    log.info("  Tumour         : %d (%.1f%%)", n_tumor,    100 * n_tumor / max(len(df), 1))
    log.info("  No tumour      : %d (%.1f%%)", n_no_tumor, 100 * n_no_tumor / max(len(df), 1))
    log.info("  Errors         : %d", len(errors))
    log.info("  Output dir     : %s", output_dir)

    if errors:
        err_df = pd.DataFrame(errors)
        err_path = output_dir / "preprocessing_errors.csv"
        err_df.to_csv(err_path, index=False)
        log.warning("Error log written to %s", err_path)


def _validate_sources(samples):
    """Quick scan of raw sources — reports missing/corrupt files."""
    ok = bad = 0
    for pid, img_path, mask_path in samples:
        try:
            img = load_image(img_path)
            assert img.ndim == 3 and img.shape[2] == 3
            if mask_path and mask_path.exists():
                mask = load_mask(mask_path)
                assert mask.ndim == 2
            ok += 1
        except Exception as exc:
            log.error("INVALID %s/%s: %s", pid, img_path.name, exc)
            bad += 1
    log.info("Validation complete: %d OK, %d FAILED", ok, bad)


# cli

def parse_args():
    p = argparse.ArgumentParser(description="NeuroVision preprocessing pipeline")
    p.add_argument("--input_dir",     required=True,  help="Root directory of raw dataset")
    p.add_argument("--output_dir",    required=True,  help="Destination for processed data")
    p.add_argument("--img_size",      type=int, default=DEFAULT_IMG_SIZE,
                   help=f"Target image size in pixels (default {DEFAULT_IMG_SIZE})")
    p.add_argument("--validate_only", action="store_true",
                   help="Only validate source files, do not write output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        size=args.img_size,
        validate_only=args.validate_only,
    )
