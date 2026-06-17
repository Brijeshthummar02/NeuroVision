import cv2
import numpy as np


def analyze_tumor_shape(mask_path):
    """
    Analyze tumor shape complexity.
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

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            "number_of_contours": 0,
            "largest_area": 0,
            "largest_perimeter": 0,
            "circularity": 0
        }

    # Largest contour
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(largest_contour)

    perimeter = cv2.arcLength(
        largest_contour,
        True
    )

    # Circularity formula
    if perimeter == 0:
        circularity = 0
    else:
        circularity = (
            4 * np.pi * area
        ) / (perimeter ** 2)

    return {
        "number_of_contours": len(contours),
        "largest_area": round(area, 2),
        "largest_perimeter": round(perimeter, 2),
        "circularity": round(circularity, 4)
    }


if __name__ == "__main__":

    mask_path = "../images/segmentation_predictions.png"

    results = analyze_tumor_shape(mask_path)

    print("\n===== TUMOR SHAPE ANALYSIS =====")

    print(
        f"Detected Contours: "
        f"{results['number_of_contours']}"
    )

    print(
        f"Largest Tumor Area: "
        f"{results['largest_area']}"
    )

    print(
        f"Largest Tumor Perimeter: "
        f"{results['largest_perimeter']}"
    )

    print(
        f"Circularity Score: "
        f"{results['circularity']}"
    )