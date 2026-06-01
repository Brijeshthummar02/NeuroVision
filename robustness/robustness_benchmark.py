import os
import cv2
import json
import numpy as np

from corruptions import (
    add_gaussian_noise,
    apply_gaussian_blur,
    adjust_brightness,
    reduce_resolution,
    add_compression_artifacts
)

# MRI image path
IMAGE_PATH = "../images/segmentation_predictions.png"

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
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError("Could not load MRI image")


print("\n===== CLEAN MRI TEST =====")

clean_result = simulate_prediction(image)

print("Clean Confidence:", clean_result["confidence"])


print("\n===== ROBUSTNESS TESTING =====")

results = []

# Gaussian Noise
results.append(
    evaluate_corruption(
        "gaussian_noise",
        add_gaussian_noise(image)
    )
)

# Blur
results.append(
    evaluate_corruption(
        "blur",
        apply_gaussian_blur(image)
    )
)

# Brightness
results.append(
    evaluate_corruption(
        "brightness",
        adjust_brightness(image)
    )
)

# Low Resolution
results.append(
    evaluate_corruption(
        "low_resolution",
        reduce_resolution(image)
    )
)

# Compression Artifacts
results.append(
    evaluate_corruption(
        "compression",
        add_compression_artifacts(image)
    )
)


print("\n===== RESULTS =====")

final_results = []

for result in results:
    confidence_drop = (
        clean_result["confidence"]
        - result["confidence"]
    )

    robustness_score = (
    result["confidence"]
    / clean_result["confidence"]
    )

    benchmark_result = {
        "corruption": result["corruption"],
        "clean_confidence": clean_result["confidence"],
        "corrupted_confidence": result["confidence"],
        "confidence_drop": round(confidence_drop, 4),
        "robustness_score": round(robustness_score, 4)
    }

    final_results.append(benchmark_result)

    print(f"""
        Corruption: {result['corruption']}
        Clean Confidence: {clean_result['confidence']}
        Corrupted Confidence: {result['confidence']}
        Confidence Drop: {round(confidence_drop, 4)}
    """)


# Save report
report_path = "reports/robustness_report.json"

with open(report_path, "w") as f:
    json.dump(final_results, f, indent=4)

print(f"\nReport saved to: {report_path}")