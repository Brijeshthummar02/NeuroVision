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