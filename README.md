# NeuroVision v2.0

Brain tumor detection and segmentation from MRI scans using a two-stage deep learning pipeline — EfficientNetB4 for classification, Attention ResUNet with CBAM + ASPP for segmentation.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.x-green?logo=flask)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25+-success)
![Dice](https://img.shields.io/badge/Dice_Score-0.94+-blue)

> ⚠️ **Research use only.** This project is not intended for clinical diagnosis. It has not undergone FDA/CE validation or regulatory review.

---

## Table of Contents

- [Overview](#overview)
- [What's New in v2.0](#whats-new-in-v20)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training](#training)
- [Inference](#inference)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)
- [References](#references)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

TumorVision processes MRI scans through two sequential stages:

1. **Classification** — EfficientNetB4 determines whether a tumor is present. If no tumor is detected, inference stops here.
2. **Segmentation** — If a tumor is found, Attention ResUNet v2.0 generates a precise pixel-level mask with boundary confidence scoring.

This two-stage design avoids running the heavier segmentation model on healthy scans, which meaningfully reduces inference time in practice.

---

## What's New in v2.0

| Component | v1.0 | v2.0 |
|-----------|------|------|
| Classification backbone | ResNet-50 | EfficientNetB4 + SE Attention |
| Segmentation model | Basic ResUNet | Attention ResUNet + CBAM + ASPP |
| Loss functions | Focal Tversky | Unified Focal + Boundary-Aware |
| Data augmentation | Basic flips/rotations | 15 medical imaging-specific augmentations |
| Inference | Standard | TTA + XLA JIT compilation |
| Classification accuracy | 97.92% | 99%+ |
| Segmentation Dice score | 0.91 | 0.94+ |
| Inference time | ~100ms | ~45ms |

---

## How It Works

### Pipeline

```
MRI Input (256×256×3)
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 1: Classification            │
│                                     │
│  EfficientNetB4 (ImageNet)          │
│  + Squeeze-and-Excitation Attention │
│  → Dense(512) → Dense(256)          │
│  → Dense(128) → Dense(64)           │
│  → Softmax(2)                       │
│                                     │
│  ~19M parameters                    │
└─────────────────────────────────────┘
         │
    Tumor found?
    ┌────┴────┐
   No        Yes
    │          │
   Done        ▼
         ┌─────────────────────────────────────┐
         │  Stage 2: Segmentation              │
         │                                     │
         │  Attention ResUNet v2.0             │
         │  Encoder: 32→64→128→256→512        │
         │  + CBAM at each encoder level       │
         │  + ASPP in bottleneck               │
         │  + Attention-gated skip connections │
         │  + SE blocks in decoder             │
         │                                     │
         │  ~2.5M parameters                   │
         └─────────────────────────────────────┘
                    │
                    ▼
         Tumor mask (256×256)
         with boundary confidence
```

### Attention Mechanisms

| Mechanism | Where | What it does |
|-----------|-------|--------------|
| CBAM | Each encoder/decoder level | Channel + spatial attention |
| SE Block | Classification head | Channel recalibration |
| Attention Gates | Skip connections | Suppresses irrelevant activations |
| ASPP | Bottleneck | Multi-scale context aggregation |

### Loss Functions

```python
# Segmentation — combined loss
Loss = 0.5 × Focal_Tversky + 0.3 × Dice + 0.2 × BCE

# Focal Tversky — handles foreground/background imbalance
Tversky = (TP + ε) / (TP + α·FN + (1-α)·FP + ε)
Focal_Tversky = (1 - Tversky)^γ
# α = 0.7 → penalizes false negatives more heavily (right call for medical imaging)
# γ = 0.75 → focuses training on hard examples

# Boundary-aware loss — sharpens edge prediction
Boundary_Loss = BCE × Edge_Weight_Map

# Unified Focal loss — best on imbalanced sets
UFC = δ × Focal_Tversky + (1-δ) × Focal_CE
```

### Data Augmentation

All augmentations are applied during training only, tuned specifically for MRI characteristics.

| Augmentation | Probability | Purpose |
|--------------|-------------|---------|
| Horizontal flip | 0.5 | Left-right invariance |
| Vertical flip | 0.5 | Orientation invariance |
| RandomRotate90 | 0.5 | Rotational invariance |
| ShiftScaleRotate | 0.5 | Position and scale variation |
| Elastic transform | 0.3 | Soft tissue deformation |
| Grid distortion | 0.3 | Shape variation |
| Optical distortion | 0.3 | Lens effect simulation |
| CLAHE | 0.5 | Contrast normalization |
| RandomBrightnessContrast | 0.5 | Intensity variation |
| RandomGamma | 0.5 | Gamma correction |
| Gaussian noise | 0.3 | Scanner noise robustness |
| Gaussian blur | 0.3 | Smoothing artifacts |
| Motion blur | 0.3 | Patient motion artifacts |
| Sharpen | 0.3 | Edge enhancement |
| Coarse dropout | 0.3 | Regularization |

---

## Model Performance

### Classification (EfficientNetB4 + SE Attention)

| Metric | v1.0 (ResNet-50) | v2.0 (EfficientNetB4) |
|--------|------------------|-----------------------|
| Accuracy | 97.92% | **99%+** |
| Precision | 0.98 | **0.99** |
| Recall | 0.98 | **0.99** |
| F1-Score | 0.98 | **0.99** |
| AUC-ROC | 0.98 | **0.995** |
| Inference time | ~100ms | **~45ms** |

### Segmentation (Attention ResUNet v2.0)

| Metric | v1.0 (ResUNet) | v2.0 (Attention ResUNet) |
|--------|----------------|--------------------------|
| Dice coefficient | 0.91 | **0.94+** |
| IoU (Jaccard) | 0.88 | **0.91+** |
| Tversky index | 0.92 | **0.95+** |
| Sensitivity | 0.93 | **0.96+** |
| Specificity | 0.98 | **0.99** |
| Boundary accuracy | — | **95%+** |

### Where the improvements come from

| Change | Effect |
|--------|--------|
| CBAM attention | +3–5% localization accuracy |
| ASPP module | Better detection of small and large tumors |
| Attention gates | Sharper tumor boundaries |
| Boundary-aware loss | Precise edge delineation |
| TTA inference | More stable predictions across scan orientations |
| Mixed precision (FP16) | 2× faster training with no accuracy loss |

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| Source | TCGA (The Cancer Genome Atlas) |
| Total scans | 3,929 |
| Patients | 110 |
| Format | TIF, 256×256 |
| Split | 70% train / 15% val / 15% test |
| Class balance | ~50% tumor / ~50% healthy |

### Download

The dataset and pre-trained model weights are hosted on Google Drive. Pick whichever works for you:

| Option | Link |
|--------|------|
| Zipped (single file) | [Download ZIP](https://drive.google.com/file/d/1FsuQvdxakt4AYcjA_D06la4fJ2kKxA9E/view?usp=sharing) |
| Unzipped (folder) | [Open Folder](https://drive.google.com/drive/folders/13_EYULCrG8GIAvfSmaWXzFhIvfn_udYh?usp=sharing) |

The ZIP contains both the MRI scan directories and the trained `.keras`/`.hdf5` weight files.

### Placing the data

After downloading and extracting, drop all `TCGA_*` folders directly into the root of the repository. The expected layout is:

```
TumorVision-2StageAI/
├── app.py
├── index.ipynb
├── data_mask.csv
├── TCGA_CS_4941_19960909/     ← TCGA folders go here
├── TCGA_CS_4942_19970222/
├── TCGA_CS_4943_20000902/
│   ...                        ← (110 patient folders total)
└── TCGA_HT_A616_19991226/
```

The training notebook reads scan paths relative to the project root, so the folder names and location need to match exactly.

---

## Project Structure

```
TumorVision-2StageAI/
│
├── app.py                            # Flask web app entry point
├── index.ipynb                       # Training notebook (v2.0)
├── utilities.py                      # All model code and helpers
│   ├── Loss functions                # Focal Tversky, Boundary-Aware, Unified Focal
│   ├── Metrics                       # Dice, IoU, Sensitivity, Specificity
│   ├── Data generators               # Augmentation pipeline
│   ├── Model architectures           # Attention ResUNet, CBAM, ASPP
│   └── TTA prediction                # Test-time augmentation helpers
│
├── classifier-enhanced-best.keras    # v2.0 classification weights
├── AttentionResUNet-v2-weights.keras # v2.0 segmentation weights
├── weights.hdf5                      # v1.0 classification weights
├── weights_seg.hdf5                  # v1.0 segmentation weights
│
├── data_mask.csv                     # Dataset labels
├── test_tumor_detection.py           # Unit tests
├── requirements-web.txt              # Python dependencies
├── .env.example                      # Environment variable template
│
├── templates/                        # Flask HTML templates
├── static/                           # CSS, JS, images
└── TCGA_*/                           # MRI scan directories (one per patient)
```

---

## Setup & Installation

**Prerequisites:** Python 3.8+, pip, git

### 1. Clone the repository

```bash
git clone https://github.com/Brijeshthummar02/TumorVision-2StageAI.git
cd TumorVision-2StageAI
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements-web.txt
```

### 4. Configure environment variables

```bash
# macOS/Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Open `.env` and fill in the following:

```env
FLASK_SECRET_KEY=your-long-random-secret-key

# Cloudinary — used for image upload and storage
# Get these from https://cloudinary.com/
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret

# MongoDB Atlas (optional)
# If omitted, the app falls back to local JSON storage
MONGO_URI=mongodb+srv://...
```

> Never commit your `.env` file. It's already in `.gitignore`.

### 5. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Training

Open `index.ipynb` in Jupyter and run all cells. The notebook handles both training stages sequentially.

### Classification model (EfficientNetB4)

Training runs in two phases — first with the backbone frozen, then with the top layers unfrozen for fine-tuning.

| Parameter | Phase 1 (frozen backbone) | Phase 2 (fine-tuning) |
|-----------|---------------------------|-----------------------|
| Backbone | EfficientNetB4 (frozen) | Top 100 layers unfrozen |
| Learning rate | 0.0001 | 0.00001 |
| Epochs | 30–50 | 20–30 |
| Optimizer | Adam | Adam (clipnorm=1.0) |
| Loss | CCE (label_smoothing=0.1) | Same |
| Batch size | 16 | 16 |

### Segmentation model (Attention ResUNet v2.0)

| Parameter | Value |
|-----------|-------|
| Encoder filters | 32 → 64 → 128 → 256 → 512 |
| Optimizer | Adam (lr=0.0001) |
| Loss | 0.5×Focal_Tversky + 0.3×Dice + 0.2×BCE |
| Epochs | 80–100 |
| LR schedule | Cosine annealing with warm restarts |
| Early stopping | patience=20, monitor=val_dice |
| Batch size | 16 |

Trained weights are saved to `classifier-enhanced-best.keras` and `AttentionResUNet-v2-weights.keras` automatically.

---

## Inference

### With Test Time Augmentation (recommended)

```python
import tensorflow as tf
from utilities import prediction

# Load models
model = tf.keras.models.load_model('classifier-enhanced-best.keras')
model_seg = tf.keras.models.load_model('AttentionResUNet-v2-weights.keras')

# Run prediction with TTA enabled
image_ids, masks, has_mask = prediction(test_df, model, model_seg, use_tta=True)
```

TTA runs inference over multiple augmented versions of each scan and averages the results. It adds a small overhead but meaningfully improves prediction stability — especially on edge cases.

### Without TTA (faster)

```python
image_ids, masks, has_mask = prediction(test_df, model, model_seg, use_tta=False)
```

---

## References

- Tan & Le (2019) — [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- Woo et al. (2018) — [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- He et al. (2015) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Ronneberger et al. (2015) — [U-Net: CNNs for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Oktay et al. (2018) — [Attention U-Net](https://arxiv.org/abs/1804.03999)
- Abraham & Khan (2018) — [Focal Tversky Loss](https://arxiv.org/abs/1810.07842)
- Hu et al. (2017) — [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- Chen et al. (2018) — [DeepLabV3+ / ASPP](https://arxiv.org/abs/1802.02611)

---

## Contributors

<a href="https://github.com/Ashwitha-Ramesh"><img src="https://github.com/Ashwitha-Ramesh.png" width="48px" title="Ashwitha-Ramesh" style="border-radius:50%;margin:4px;" /></a>
<a href="https://github.com/LIANNAKA"><img src="https://github.com/LIANNAKA.png" width="48px" title="LIANNAKA" style="border-radius:50%;margin:4px;" /></a>
<a href="https://github.com/Sujan075"><img src="https://github.com/Sujan075.png" width="48px" title="Sujan075" style="border-radius:50%;margin:4px;" /></a>
<a href="https://github.com/bhavyapandiya29"><img src="https://github.com/bhavyapandiya29.png" width="48px" title="bhavyapandiya29" style="border-radius:50%;margin:4px;" /></a>
<a href="https://github.com/itsdakshjain"><img src="https://github.com/itsdakshjain.png" width="48px" title="itsdakshjain" style="border-radius:50%;margin:4px;" /></a>
<a href="https://github.com/jamunatg2006-sys"><img src="https://github.com/jamunatg2006-sys.png" width="48px" title="jamunatg2006-sys" style="border-radius:50%;margin:4px;" /></a>
<a href="https://github.com/kadiashailee-devias"><img src="https://github.com/kadiashailee-devias.png" width="48px" title="kadiashailee-devias" style="border-radius:50%;margin:4px;" /></a>
<a href="https://github.com/shambavi2007"><img src="https://github.com/shambavi2007.png" width="48px" title="shambavi2007" style="border-radius:50%;margin:4px;" /></a>

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---
<div align="center">

**⭐ Star this repo if you found it useful!**

Made with ❤️ for advancing medical AI

</div>
