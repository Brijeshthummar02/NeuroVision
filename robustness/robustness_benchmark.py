import os
import cv2
import json
import numpy as np
import glob
from corruptions import (
    add_gaussian_noise,
    apply_gaussian_blur,
    adjust_brightness,
    reduce_resolution,
    add_compression_artifacts
)

# MRI image path
DATASET_PATH = "MRI Datasets"

# Output directory
OUTPUT_DIR = "benchmark_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
def simulate_prediction(image):
    """
    Temporary mock prediction function.
    Later this can be replaced with real model inference.
    """

    # Simulate confidence using image quality
    variance = np.var(image)

    mean_intensity = np.mean(image)

    quality_score = (
        (variance / 10000)
        + (mean_intensity / 255)
    ) / 2

    confidence = max(
        0.50,
        min(0.99, quality_score)
    )

    return {
        "has_tumor": True,
        "confidence": round(confidence, 4)
    }


def evaluate_corruption(name, corrupted_image):
    """
    Evaluate one corrupted MRI image.
    """

    output_path = f"{OUTPUT_DIR}/{name}.jpg"

    cv2.imwrite(output_path, corrupted_image)

    result = simulate_prediction(corrupted_image)

    return {
        "corruption": name,
        "confidence": result["confidence"]
    }


# Load original MRI image
image_paths = glob.glob(
    os.path.join(DATASET_PATH, "**", "*.tif"),
    recursive=True
)

image_paths = [
    p for p in image_paths
    if "_mask.tif" not in p
]

dataset_results = []

print("\n===== DATASET ROBUSTNESS TEST =====")
print("Total MRI scans:", len(image_paths))

for image_path in image_paths:

    image = cv2.imread(image_path)

    if image is None:
        continue

    clean_result = simulate_prediction(image)

    results = []
for image_path in image_paths:

    image = cv2.imread(image_path)

    if image is None:
        continue

    clean_result = simulate_prediction(image)

    corruptions = {
        "gaussian_noise": add_gaussian_noise(image),
        "blur": apply_gaussian_blur(image),
        "brightness": adjust_brightness(image),
        "low_resolution": reduce_resolution(image),
        "compression": add_compression_artifacts(image)
    }

    for corruption_name, corrupted_image in corruptions.items():

        result = simulate_prediction(corrupted_image)

        confidence_drop = (
            clean_result["confidence"]
            - result["confidence"]
        )

        dataset_results.append({
            "corruption": corruption_name,
            "confidence_drop": confidence_drop
        })

summary = {}

for item in dataset_results:

    corruption = item["corruption"]

    if corruption not in summary:
        summary[corruption] = []

    summary[corruption].append(
        item["confidence_drop"]
    )

final_results = []

for corruption, values in summary.items():

    avg_drop = sum(values) / len(values)

    final_results.append({
        "corruption": corruption,
        "average_confidence_drop": round(avg_drop, 4)
    })
os.makedirs("robustness/reports", exist_ok=True)
report_path = "robustness/reports/robustness_report.json"

with open(report_path, "w") as f:
    json.dump(final_results, f, indent=4)

print(final_results)
print(f"\nReport saved to: {report_path}")