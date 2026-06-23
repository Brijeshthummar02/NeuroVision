from analysis.tumor_metrics import calculate_tumor_metrics
from analysis.severity_analysis import classify_severity
from analysis.region_analysis import detect_regions
from analysis.shape_analysis import analyze_shape

def run_tumor_analysis(mask_binary):
    metrics = calculate_tumor_metrics(mask_binary)
    severity = classify_severity(metrics["tumor_percentage"])
    regions = detect_regions(mask_binary)
    shape = analyze_shape(mask_binary)

    return {
        **metrics,
        "severity":severity,
        **regions,
        **shape,
        "comparison_ready":True
    }