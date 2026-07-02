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

| Attribute | BRISC 2025 | BraTS 2021 | Mendeley 4-class | Brain Tumor MRI Dataset: Segmentation & Classification | UCSF-PDGM |
|---|---|---|---|---|---|
| Dataset link | [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) | [TCIA](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/) | [Mendeley](https://data.mendeley.com/datasets/zwr4ntf94j/6) | [Kaggle](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification) | [TCIA](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) |
| Total scans | 6,000 T1-weighted MRI slices (5,000 train / 1,000 test) | Multi-parametric dataset (T1, T1Gd, T2, T2-FLAIR) | 11,148 T1-weighted MRI images | ~5,000 total images; ~2,700 with segmentation masks |
| Patients | Not specified (multi-source curated) | 1,480 publicly available subjects | Not specified | Not specified | 495 |
| Tumor types | Glioma, Meningioma, Pituitary, No Tumor | Glioma only (HGG + LGG) | Glioma, Meningioma, Pituitary, No Tumor | No Tumor, Glioma, Meningioma, Pituitary | Diffuse glioma only (GBM + LGG) |
| Image format | JPG (images), PNG (masks) | DICOM and NIfTI (.nii.gz), 3D volumetric | JPEG/PNG | JPEG/PNG | NIfTI (.nii.gz) - 3D volumetric, 3T scanner |
| Segmentation masks | Yes (pixel-wise, radiologist-verified) | Yes (manually generated labels for enhancing tumor, necrosis, and edema) | No (classification only) | Yes (binary or multi-class mask aligned with the image) | Yes (enhancing tumor, non-enhancing/necrotic, FLAIR abnormality) |
| Source institution | CV Lab SHUT (multi-source curated) | Multi-institutional (15+ hospitals globally) | National Institute of Textile Engineering and Research, Dhaka | Curated and enhanced from Kaggle Brain Tumor MRI Dataset and SciDB Brain Tumor Dataset | University of California San Francisco (single institution) |
| License | CC BY 4.0 | CC BY 4.0 + data citation required | CC BY 4.0 | MIT License for enhancements/repository; original datasets under respective licenses | CC BY 4.0 + TCIA policies |
| 256x256 ready | No (resize is needed) | No (3D volumetric data) | Not specified (likely needs resize) | No (resize is needed) | No (3D volumetric) |
| Needs resize | Yes | Yes | Yes | Yes | Yes |
| Needs format conversion | Not required | Yes (NIfTI/DICOM to TIF, 3D to 2D extraction) | Not required | Not required | Yes (NIfTI to TIF, 3D to 2D extraction) |
| Needs label conversion | No (4-class labels match NeuroVision) | Yes (multi-region labels to binary mask) | No (4-class labels match NeuroVision) | No (label mapping already provided: 0 No Tumor, 1 Glioma, 2 Meningioma, 3 Pituitary) | Yes (3 compartments to binary mask) |
| Recommendation | Recommended for both classification and segmentation (with preprocessing) | Recommended for segmentation only (heavy preprocessing required) | Recommended for classification only | Recommended for both classification and segmentation | Lower priority (heavy preprocessing, glioma only, overlapping with BraTS) |

---

## Summary

| Dataset | Use Case | Preprocessing Effort |
|---|---|---|
| **BRISC 2025** | Both classification and segmentation | Moderate (resize to 256x256) |
| **BraTS 2021** | Segmentation only | Heavy (3D to 2D extraction, DICOM/NIfTI to TIF/jpg/png, resize, label conversion) |
| **Mendeley 4-class** | Classification only | (resize to 256x256, standard folder structure) |
| **Brain Tumor MRI Dataset: Segmentation & Classification** | Both classification and segmentation | Moderate ( resize to 256x256, use provided masks and train/val folders) |
| **UCSF-PDGM** | Segmentation / glioma only | Heavy (3D NIfTI processing, label conversion, format conversion) |

---

## Final Recommended Priority

### Priority 1:
1. BRISC 2025
2. Brain Tumor MRI Dataset: Segmentation & Classification

### Priority 2:
3. Mendeley 4-Class
4. Brats 2021 (segmentation only and preprocessing is needed)

### Lower Priority:
5. UCSF-PDGM (high-quality dataset but requires substantial preprocessing and overlaps with BraTS for glioma segmentation tasks)

---

## License Summary

| Dataset | License Details |
|---|---|
| BRISC 2025 | CC BY 4.0 (attribution required; citation of accompanying publication recommended) |
| BraTS 2021 | CC BY 4.0; data citation required through TCIA |
| Mendeley 4-class | CC BY 4.0 (attribution required, commercial use allowed) |
| Brain Tumor MRI Dataset: Segmentation & Classification | Citing of the original sources is required |
| UCSF-PDGM | CC BY 4.0 + TCIA data usage policies (no patient identification, acknowledge dataset in publications) |

---

## Experimental Results 

The experiments were performed by using the original TCGA dataset as the baseline, followed by progressively integrating BRISC, Mendeley 4-class, and the Brain Tumor MRI Dataset: Segmentation & Classification. BraTS 2021 and UCSF-PDGM were not experimentally integrated due to their higher preprocessing requirements.

### Classification Results

| Metric | TCGA | TCGA + BRISC | TCGA + BRISC + Mendeley | TCGA + BRISC + Brain tumor MRI Dataset |
|----------|----------|----------|----------|----------|
| Accuracy | 0.7881 | 0.5251 | 0.6282 | 0.6249 |
| Precision | 0.7984 | 0.6489 | 0.3946 | 0.6461 |
| Recall | 0.7881 | 0.5251 | 0.6282 | 0.6249 |
| F1-Score | 0.7905 | 0.3679 | 0.4847 | 0.5057 |
| AUC-ROC | 0.8641 | 0.5145 | 0.5290 | 0.5070 |
| Inference Time (ms/image) | 15.74 | 6.87 | 4.33 | 5.17 |

### Segmentation Results

| Metric | TCGA | TCGA + BRISC | TCGA + BRISC + Mendeley | TCGA + BRISC + Brain tumor MRI Dataset |
|----------|----------|----------|----------|----------|
| Dice Coefficient | 0.8803 | 0.0356 | 0.026650 | 0.000017 |
| IoU (Jaccard) | 0.8004 | 0.0194 | 0.014286 | 0.000008 |
| Tversky Index | 0.8766 | 0.0408 | 0.029274 | 0.000012 |
| Sensitivity | 0.8745 | 0.0680 | 0.043641 | 0.000008 |
| Specificity | 0.9974 | 0.9366 | 0.972273 | 0.999899 |
| Inference Time (ms/image) | 147.62 | 193.66 | 185.68 | 193.49 |

- BRISC dataset: 
    - masks were resized to match the model input resolution.
    - only valid masks were kept in binary/single-channel form 

- Brain tumor MRI dataset: 
    - RGB or multi-channel masks were converted to grayscale before segmentation training.
    - masks were thresholded to binary form after conversion.
    -  masks were resized to 256 × 256.

- Mendeley 4-class dataset:
    - images were resized to 256 × 256.
    - the existing four-class folder structure was used directly for classification.



> **Note:** For the TCGA + BRISC + Mendeley segmentation evaluation, metrics were computed on **206 valid tumor-mask samples**.

The additional datasets did not improve the TCGA baseline performance in the current pipeline. The decrease in classification and segmentation metrics is likely due to differences in preprocessing pipelines, image formats, and dataset distributions. Additional preprocessing and dataset harmonization are required before combining these datasets for training.

---

## Notes

- Image dimensions for BRISC 2025, Mendeley 4-class, and Brain Tumor MRI Dataset: Segmentation & Classification require verification through direct dataset inspection.
- BRISC 2025 is the only dataset that supports both classification and segmentation with relatively lower preprocessing effort.
- BraTS 2021 is available through TCIA and provides a large, multi-institutional glioma dataset, but it requires substantial preprocessing before integration.
- Mendeley 4-class and Brain Tumor MRI Dataset: Segmentation & Classification are no longer redundant, since the latter also supports segmentation while Mendeley remains classification-only.