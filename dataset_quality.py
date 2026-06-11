import os
import json
import csv
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# PATHS  (all relative to project root)
CSV_PATH    = "data_mask.csv"   # already in the repo root
IMG_ROOT    = "./"              # images live as  ./TCGA_xxx/TCGA_xxx_1.tif
OUTPUT_DIR  = "dataset_reports"


# STEP 1 — Dataset Statistics
def generate_statistics(df: pd.DataFrame) -> dict:
    total   = len(df)
    counts  = df["mask"].value_counts().to_dict()

    # per-patient breakdown
    patient_per_class = (
        df.groupby("mask")["patient_id"].nunique().to_dict()
    )

    # data-integrity checks
    duplicate_images   = int(df["image_path"].duplicated().sum())
    empty_image_paths  = int(df["image_path"].str.strip().eq("").sum())
    invalid_mask_vals  = int((~df["mask"].isin([0, 1])).sum())

    stats = {
        "total_samples"        : total,
        "unique_patients"      : int(df["patient_id"].nunique()),
        "mask_distribution"    : {str(k): int(v) for k, v in counts.items()},
        "patients_per_class"   : {str(k): int(v) for k, v in patient_per_class.items()},
        "duplicate_image_paths": duplicate_images,
        "empty_image_path_rows": empty_image_paths,
        "invalid_mask_values"  : invalid_mask_vals,
    }

    print("\n── Dataset Statistics ──────────────────────────")
    print(f"  Total samples      : {total:,}")
    print(f"  Unique patients    : {stats['unique_patients']}")
    print(f"  mask=0 (No Tumor)  : {counts.get(0, 0):,}")
    print(f"  mask=1 (Tumor)     : {counts.get(1, 0):,}")
    print(f"  Duplicate paths    : {duplicate_images}")
    print(f"  Empty path rows    : {empty_image_paths}")
    print(f"  Invalid mask values: {invalid_mask_vals}")

    return stats


# STEP 2 — Filter Corrupted Samples
def _is_corrupted(img_path: str, mask_path: str) -> tuple[bool, str]:
    """
    Returns (is_corrupted, reason).
    Checks: file exists, OpenCV can read it, image is not blank. 
    """ 
    for label, path in [("image", img_path), ("mask", mask_path)]:
        full = os.path.join(IMG_ROOT, path)
        if not os.path.isfile(full):
            return True, f"{label}_not_found"
        img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True, f"{label}_unreadable"
        if label == "image" and img.std() < 1.0:
            return True, f"{label}_blank"
    return False, "" 


def filter_corrupted(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Scans every row. Returns (clean_df, list_of_removed_records).
    NOTE: If the actual image folder is not present locally (e.g. you only
    have the CSV), file-not-found rows are flagged but the clean CSV still
    keeps them — set SKIP_MISSING=True below to remove them too.
    """
    SKIP_MISSING = False   # set True if you want to drop missing-file rows

    removed = []
    keep    = []

    print("\n── Scanning for Corrupted Samples ──────────────")
    print(f"  Checking {len(df):,} rows …")

    for idx, row in df.iterrows():
        corrupted, reason = _is_corrupted(row["image_path"], row["mask_path"])
        if corrupted:
            if reason.endswith("_not_found") and not SKIP_MISSING:
                keep.append(idx)           # keep row, just log it
            else:
                removed.append({
                    "index"     : int(idx),
                    "image_path": row["image_path"],
                    "reason"    : reason,
                })
        else:
            keep.append(idx)

    clean_df = df.loc[keep].reset_index(drop=True)

    print(f"  Rows removed (corrupted) : {len(removed)}")
    print(f"  Rows kept (clean)        : {len(clean_df):,}")

    return clean_df, removed


# STEP 3 — Class Imbalance Report
def class_imbalance_report(df: pd.DataFrame) -> dict:
    counts = df["mask"].value_counts().to_dict()
    total  = len(df)

    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)

    ratio = round(max(n0, n1) / min(n0, n1), 3) if min(n0, n1) > 0 else float("inf")

    # sklearn-style class_weight='balanced' formula
    cw0 = round(total / (2 * n0), 4) if n0 else None
    cw1 = round(total / (2 * n1), 4) if n1 else None

    if ratio < 1.5:
        severity    = "LOW"
        suggestions = ["Dataset is well-balanced. No special action required."]
    elif ratio < 3.0:
        severity    = "MODERATE"
        suggestions = [
            f"Pass class_weight={{0: {cw0}, 1: {cw1}}} to model.fit().",
            "Apply random oversampling on the minority class.",
            "Use stratified train/val/test splits (stratify=df['mask']).",
            "Evaluate with F1-score / PR-AUC, not just accuracy.",
        ]
    else:
        severity    = "SEVERE"
        suggestions = [
            "Use focal loss instead of binary cross-entropy.",
            "Oversample minority class with augmentation (flip, rotate, crop).",
            "Undersample majority class (random or Tomek links).",
            "Use stratified k-fold cross-validation.",
        ]

    report = {
        "class_summary": {
            "0": {"label": "No Tumor", "count": n0,
                  "percentage": round(100 * n0 / total, 2)},
            "1": {"label": "Tumor Present", "count": n1,
                  "percentage": round(100 * n1 / total, 2)},
        },
        "imbalance_ratio"          : ratio,
        "severity"                 : severity,
        "recommended_class_weights": {"0": cw0, "1": cw1},
        "suggestions"              : suggestions,
    }

    print("\n── Class Imbalance Report ──────────────────────")
    print(f"  mask=0 (No Tumor)    : {n0:,}  ({100*n0/total:.1f}%)")
    print(f"  mask=1 (Tumor)       : {n1:,}  ({100*n1/total:.1f}%)")
    print(f"  Imbalance ratio      : {ratio}:1  [{severity}]")
    print(f"  Recommended weights  : {{0: {cw0}, 1: {cw1}}}")
    print("  Suggestions:")
    for s in suggestions:
        print(f"    → {s}")

    return report


# STEP 4 — Plot Distribution Chart
def plot_distribution(counts: dict, out_path: str):
    labels = ["No Tumor\n(mask=0)", "Tumor Present\n(mask=1)"]
    values = [counts.get("0", 0), counts.get("1", 0)]
    colors = ["#4CAF50", "#F44336"]
    total  = sum(values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("NeuroVision — Class Distribution", fontsize=14, fontweight="bold")

    # Bar
    bars = axes[0].bar(labels, values, color=colors, edgecolor="white",
                       linewidth=1.5, width=0.5)
    axes[0].set_title("Sample Count per Class")
    axes[0].set_ylabel("Number of Samples")
    axes[0].spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 30,
                     f"{val:,}\n({100*val/total:.1f}%)",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, max(values) * 1.2)

    # Pie
    _, _, autotexts = axes[1].pie(
        values, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    axes[1].set_title("Class Proportion")

    patches = [mpatches.Patch(color=c, label=f"{l.replace(chr(10), ' ')}: {v:,}")
               for c, l, v in zip(colors, labels, values)]
    axes[1].legend(handles=patches, loc="lower center",
                   bbox_to_anchor=(0.5, -0.12), ncol=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out_path}")


# MAIN
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load CSV ──────────────────────────────
    print(f"\nLoading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH)
    # ensure mask column is int for comparisons
    df["mask"] = df["mask"].astype(int)

    # ── Step 1: Statistics ────────────────────
    stats = generate_statistics(df)
    stats_path = os.path.join(OUTPUT_DIR, "dataset_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  ✓ Saved → {stats_path}")

    # ── Step 2: Filter corrupted ──────────────
    clean_df, removed = filter_corrupted(df)
    clean_csv_path = os.path.join(OUTPUT_DIR, "data_mask_clean.csv")
    clean_df.to_csv(clean_csv_path, index=False)
    print(f"  ✓ Saved → {clean_csv_path}  ({len(clean_df):,} rows)")

    if removed:
        removed_path = os.path.join(OUTPUT_DIR, "corrupted_samples.json")
        with open(removed_path, "w") as f:
            json.dump(removed, f, indent=2)
        print(f"  ✓ Corrupted list → {removed_path}")

    # ── Step 3: Imbalance report ──────────────
    imb = class_imbalance_report(clean_df)
    imb_path = os.path.join(OUTPUT_DIR, "class_imbalance_report.json")
    with open(imb_path, "w") as f:
        json.dump(imb, f, indent=2)
    print(f"\n  ✓ Saved → {imb_path}")

    # ── Step 4: Chart ─────────────────────────
    chart_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plot_distribution(
        {str(k): v for k, v in clean_df["mask"].value_counts().to_dict().items()},
        chart_path,
    )

    print(f"\n✅  All outputs saved to  ./{OUTPUT_DIR}/\n") 


if __name__ == "__main__": 
    main() 