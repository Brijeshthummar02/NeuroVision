# Brain MRI Dataset Candidates for NeuroVision Expansion

## Current Dataset Baseline (Reference)

| Attribute | Detail |
|---|---|
| Source | TCGA (The Cancer Genome Atlas) via LGG Segmentation |
| Total scans | 3,929 |
| Patients | 110 |
| Format | TIF, 256x256 |
| Segmentation masks | Yes (binary FLAIR masks) |
| License | CC BY 3.0 |

---

## Candidate Datasets Comparison

| Attribute | BRISC 2025 | BraTS 2021 | Mendeley 4-class | MRI-BT | UCSF-PDGM |
|---|---|---|---|---|---|
| Dataset link | [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) | [TCIA](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/) | [Mendeley](https://data.mendeley.com/datasets/zwr4ntf94j/6) | [Kaggle](https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images) | [TCIA](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) |
| Total scans | 6,000 T1-weighted MRI slices (5,000 train / 1,000 test) | Multi-parametric dataset (T1, T1Gd, T2, T2-FLAIR) | 11,148 T1-weighted MRI images | 7,023 T1-weighted MRI images | 495 patients with histopathologically-confirmed diffuse gliomas  |
| Patients | Not specified (multi-source curated) | 1,480 publicly available subjects | Not specified | Not specified | 495  |
| Tumor types | Glioma, Meningioma, Pituitary, No Tumor | Glioma only (HGG + LGG) | Glioma, Meningioma, Pituitary, No Tumor | Glioma, Meningioma, Pituitary, No Tumor | Diffuse glioma only (GBM + LGG) |
| Image format | JPG (images), PNG (masks) | DICOM and NIfTI (.nii.gz), 3D volumetric | JPEG/PNG | JPEG/PNG | NIfTI (.nii.gz) - 3D volumetric, 3T scanner |
| Segmentation masks | Yes (pixel-wise, radiologist-verified) | Yes (manually generated labels for enhancing tumor, necrosis, and edema) | No (classification only) | No (classification only) | Yes (enhancing tumor, non-enhancing/necrotic, FLAIR abnormality) |
| Source institution | CV Lab SHUT (multi-source curated) | Multi-institutional (15+ hospitals globally) | National Institute of Textile Engineering and Research, Dhaka | Combined from multiple public sources (Figshare, Br35H, SARTAJ) | University of California San Francisco (single institution) |
| License | CC BY 4.0 | CC BY 4.0 + data citation required | CC BY 4.0 | MIT License | CC BY 4.0 + TCIA policies |
| 256x256 ready | No (resize is needed) | No (3D volumetric data) | Not specified (likely needs resize) | Not specified (likely needs resize) | No (3D volumetric) |
| Needs resize | Yes | Yes | Yes | Yes | Yes |
| Needs format conversion | Yes (JPG/PNG to TIF) | Yes (NIfTI/DICOM to TIF, 3D to 2D extraction) | Yes (JPEG/PNG to TIF) | Yes (JPEG/PNG to TIF) | Yes (NIfTI to TIF, 3D to 2D extraction) |
| Needs label conversion | No (4-class labels match NeuroVision) | Yes (multi-region labels to binary mask) | No (4-class labels match NeuroVision) | No (4-class labels match NeuroVision) | Yes (3 compartments to binary mask) |
| Recommendation | Recommended for both classification and segmentation (with preprocessing) | Recommended for segmentation only (heavy preprocessing required) | Recommended for classification only | Recommended for classification only (smaller alternative to Mendeley) | Lower priority (heavy preprocessing, glioma only, overlapping with BraTS) |

---

## Summary

| Dataset | Use Case | Preprocessing Effort |
|---|---|---|
| **BRISC 2025** | Both classification and segmentation | Moderate (JPG/PNG to TIF, resize to 256x256) |
| **BraTS 2021** | Segmentation only | Heavy (3D to 2D extraction, DICOM/NIfTI to TIF, resize, label conversion) |
| **Mendeley 4-class** | Classification only | Light (JPEG/PNG to TIF, resize to 256x256) |
| **MRI-BT** | Classification only (alternative to Mendeley) | Light (JPEG/PNG to TIF, resize to 256x256) |
| **UCSF-PDGM** | Segmentation / glioma only | Heavy (3D NIfTI processing, label conversion, format conversion) |

---

## Final Recommended Priority

### Priority 1:
1. BRISC 2025
2. BraTS 2021

### Priority 2:
3. Mendeley 4-Class
4. MRI-BT

### Lower Priority:
5. UCSF-PDGM (high-quality dataset but requires substantial preprocessing and overlaps with BraTS for glioma segmentation tasks)

---

## License Summary

| Dataset | License Details |
|---|---|
| BRISC 2025 | CC BY 4.0 (attribution required; citation of accompanying publication recommended) |
| BraTS 2021 | CC BY 4.0; data citation required through TCIA |
| Mendeley 4-class | CC BY 4.0 (attribution required, commercial use allowed) |
| MRI-BT | MIT License (permissive reuse with copyright notice) |
| UCSF-PDGM | CC BY 4.0 + TCIA data usage policies (no patient identification, acknowledge dataset in publications) |

---

## Notes

- Image dimensions for BRISC 2025, Mendeley 4-class, and MRI-BT require verification through direct dataset inspection.
- All recommended datasets require format conversion to TIF to match the current NeuroVision pipeline.
- BRISC 2025 is the only dataset that supports both classification and segmentation with relatively lower preprocessing effort.
- BraTS 2021 is available through TCIA and provides a large, multi-institutional glioma dataset, but it requires substantial preprocessing before integration.
- Mendeley 4-class and MRI-BT are redundant for classification; one of them is sufficient depending on preferred dataset size.