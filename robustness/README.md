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
## Benchmark Results

| Corruption | Confidence Drop |
|------------|----------------|
| Gaussian Noise | 0.0599 |
| Blur | 0.0132 |
| Brightness | 0.0013 |
| Low Resolution | 0.0117 |
| Compression | 0.0144 |

### Observations

- Gaussian noise produced the largest confidence degradation.
- Brightness changes had minimal impact.
- The benchmark demonstrates how MRI model confidence changes under image corruption.