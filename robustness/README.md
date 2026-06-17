# MRI Robustness Benchmarking

This module provides real-world robustness evaluation for MRI tumor detection systems.

## Features

- Gaussian noise testing
- Blur corruption testing
- Brightness shift testing
- Low-resolution simulation
- Compression artifact simulation
- Confidence degradation tracking
- Robustness scoring
- Automated benchmark reporting
- Visualization of corruption impact

---

## Folder Structure

robustness/
├── corruptions.py
├── robustness_benchmark.py
├── visualize_results.py
├── reports/
└── README.md

---

## Usage

Run robustness benchmark:

```bash
python robustness_benchmark.py
```
## Dataset Benchmark

Total MRI Scans Evaluated: 3954

Corruptions Evaluated:
- Gaussian Noise
- Blur
- Brightness
- Low Resolution
- Compression
## Outputs Generated:

- robustness_report.json
- robustness_plot.png
- model_comparison.md

## Observations
- Benchmarking was performed across the complete MRI dataset.
- All MRI scans were evaluated under multiple corruption conditions.
- The framework measures confidence degradation caused by common MRI image corruptions.
- Results can be used as a baseline for future robustness comparisons and model improvements.