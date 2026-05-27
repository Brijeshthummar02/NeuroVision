import cv2
import numpy as np


def calculate_tumor_metrics(mask_path):
    """
    Calculate tumor size metrics from segmentation mask.
    """

    # Load segmentation mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")

    # Convert to binary mask
    _, binary_mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Total brain pixels
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]

    # Tumor pixels
    tumor_pixels = np.count_nonzero(binary_mask)

    # Tumor occupancy percentage
    tumor_percentage = (
        tumor_pixels / total_pixels
    ) * 100

    return {
        "total_pixels": int(total_pixels),
        "tumor_pixels": int(tumor_pixels),
        "tumor_percentage": round(tumor_percentage, 2)
    }


if __name__ == "__main__":

    mask_path = "../images/segmentation_predictions.png"

    metrics = calculate_tumor_metrics(mask_path)

    print("\n===== TUMOR SIZE METRICS =====")

    print(f"Total Pixels: {metrics['total_pixels']}")

    print(f"Tumor Pixels: {metrics['tumor_pixels']}")

    print(f"Tumor Occupancy: {metrics['tumor_percentage']}%")