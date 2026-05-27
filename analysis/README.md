# Tumor Size & Severity Analysis

This module provides advanced post-processing analysis for segmented brain tumor MRI scans.

## Features

### Tumor Size Analysis
Calculates:
- tumor pixel count
- tumor occupancy percentage
- relative brain occupancy

### Severity Classification
Severity levels:
- Small
- Medium
- Large
- Critical

Based on tumor occupancy percentage.

### Multiple Region Detection
Detects:
- disconnected tumor regions
- total number of tumor regions
- largest tumor component

### Shape Complexity Analysis
Measures:
- contour detection
- tumor perimeter
- circularity score
- shape irregularity

### Unified Analysis Pipeline
The module combines all analysis stages into:
```bash
python tumor_analysis_pipeline.py