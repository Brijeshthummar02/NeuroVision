import cv2
import numpy as np


def detect_tumor_regions(mask_path):
    """
    Detect multiple disconnected tumor regions.
    """

    # Load segmentation mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")

    # Convert to binary image
    _, binary_mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    # Ignore background
    tumor_regions = num_labels - 1

    region_sizes = []

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        region_sizes.append(int(area))

    largest_region = max(region_sizes) if region_sizes else 0

    return {
        "number_of_regions": tumor_regions,
        "region_sizes": region_sizes,
        "largest_region": largest_region
    }


if __name__ == "__main__":

    mask_path = "../images/segmentation_predictions.png"

    results = detect_tumor_regions(mask_path)

    print("\n===== TUMOR REGION ANALYSIS =====")

    print(f"Number of Regions: {results['number_of_regions']}")

    print(f"Total Regions Detected: {results['number_of_regions']}")

    print(
        f"Largest Region Size: "
        f"{results['largest_region']}"
    )

    print(
        f"Top 10 Region Sizes: "
        f"{sorted(results['region_sizes'], reverse=True)[:10]}"
    )

    print(f"Largest Region: {results['largest_region']}")