from tumor_metrics import calculate_tumor_metrics


def classify_severity(tumor_percentage):
    """
    Classify tumor severity based on occupancy percentage.
    """

    if tumor_percentage < 2:
        return "Small"

    elif tumor_percentage < 8:
        return "Medium"

    elif tumor_percentage < 15:
        return "Large"

    else:
        return "Critical"


if __name__ == "__main__":

    mask_path = "../images/segmentation_predictions.png"

    metrics = calculate_tumor_metrics(mask_path)

    severity = classify_severity(
        metrics["tumor_percentage"]
    )

    print("\n===== TUMOR SEVERITY ANALYSIS =====")

    print(f"Tumor Occupancy: {metrics['tumor_percentage']}%")

    print(f"Severity Level: {severity}")