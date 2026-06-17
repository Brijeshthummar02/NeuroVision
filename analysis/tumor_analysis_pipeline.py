import json

from tumor_metrics import calculate_tumor_metrics

from severity_analysis import classify_severity

from region_detection import detect_tumor_regions

from shape_analysis import analyze_tumor_shape


def run_complete_analysis(mask_path):
    """
    Run complete tumor analysis pipeline.
    """

    # Tumor metrics
    metrics = calculate_tumor_metrics(mask_path)

    # Severity analysis
    severity = classify_severity(
        metrics["tumor_percentage"]
    )

    # Region detection
    regions = detect_tumor_regions(mask_path)

    # Shape analysis
    shape = analyze_tumor_shape(mask_path)

    return {
        "tumor_metrics": metrics,
        "severity": severity,
        "region_analysis": regions,
        "shape_analysis": shape
    }


if __name__ == "__main__":

    mask_path = "../images/segmentation_predictions.png"

    results = run_complete_analysis(mask_path)

    print("\n========== COMPLETE TUMOR ANALYSIS ==========")

    print("\n--- Tumor Metrics ---")

    print(
        f"Tumor Occupancy: "
        f"{results['tumor_metrics']['tumor_percentage']}%"
    )

    print(
        f"Tumor Pixels: "
        f"{results['tumor_metrics']['tumor_pixels']}"
    )

    print("\n--- Severity Analysis ---")

    print(
        f"Severity Level: "
        f"{results['severity']}"
    )

    print("\n--- Region Analysis ---")

    print(
        f"Total Regions: "
        f"{results['region_analysis']['number_of_regions']}"
    )

    print(
        f"Largest Region: "
        f"{results['region_analysis']['largest_region']}"
    )

    print(
        f"Top 10 Region Sizes: "
        f"{sorted(results['region_analysis']['region_sizes'], reverse=True)[:10]}"
    )

    print("\n--- Shape Analysis ---")

    print(
        f"Contours Detected: "
        f"{results['shape_analysis']['number_of_contours']}"
    )

    print(
        f"Circularity Score: "
        f"{results['shape_analysis']['circularity']}"
    )

    # Save report
    with open(
        "reports/tumor_analysis_report.json",
        "w"
    ) as f:

        json.dump(results, f, indent=4)

    print(
        "\nTumor analysis report saved "
        "to reports/tumor_analysis_report.json"
    )