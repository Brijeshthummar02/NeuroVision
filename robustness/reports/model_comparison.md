# MRI Model Benchmark Comparison

## Baseline Results

Clean MRI Confidence: 0.7862

## Robustness Results

| Corruption | Confidence Drop |
|------------|----------------|
| Gaussian Noise | 0.0599 |
| Blur | 0.0132 |
| Brightness | 0.0013 |
| Low Resolution | 0.0117 |
| Compression | 0.0144 |

## Findings

- Gaussian noise causes the largest confidence degradation.
- Brightness changes have minimal impact.
- Model remains relatively stable under blur and compression.
- Robustness benchmarking framework successfully evaluates confidence degradation under common MRI corruptions.

## Generated Outputs

- robustness_report.json
- robustness_plot.png