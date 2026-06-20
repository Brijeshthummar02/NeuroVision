# 🧠 NeuroScan AI - Brain Tumor Detection Web Application

<div align="center">

![NeuroScan AI](https://img.shields.io/badge/NeuroScan-AI-334EAC?style=for-the-badge&logo=brain&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)

**Advanced AI-Powered Brain Tumor Detection System**

A beautiful, modern web interface for real-time MRI scan analysis using deep learning.

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack)

</div>

---

## 🎯 Overview

**NeuroScan AI** is a cutting-edge web application that leverages deep learning to detect and localize brain tumors in MRI scans. Built for the hackathon, it features a stunning, minimalist UI with a custom color palette and provides instant AI-powered analysis.

### ✨ Key Highlights

- 🚀 **Lightning Fast**: Analyze MRI scans in under 3 seconds
- 🎯 **95.8% Accuracy**: ResNet-50 classification model
- 🔍 **Pixel-Perfect Segmentation**: ResUNet architecture for precise tumor localization
- 🎨 **Beautiful UI/UX**: Clean, minimalist design with custom color palette
- 📱 **Fully Responsive**: Works seamlessly on desktop, tablet, and mobile
- 🔒 **Secure**: File validation and size limits

---

## 🌟 Features

### Frontend Features
- ✅ **Drag & Drop Upload**: Intuitive file upload with drag-and-drop support
- ✅ **Real-time Preview**: Instant image preview before analysis
- ✅ **Animated Loading**: Beautiful loading states with progress indicators
- ✅ **Interactive Results**: Visual comparison of original, mask, and overlay
- ✅ **Smooth Animations**: Polished transitions and scroll effects
- ✅ **Responsive Design**: Mobile-first, works on all devices
- ✅ **Dark/Light Themes**: Custom color palette inspired by space

### Backend Features
- ✅ **Two-Stage Pipeline**: Classification → Segmentation
- ✅ **RESTful API**: Clean, documented endpoints
- ✅ **Model Loading**: Automatic pre-trained model initialization
- ✅ **Image Processing**: Advanced preprocessing and normalization
- ✅ **Error Handling**: Comprehensive error messages
- ✅ **Health Checks**: API status monitoring

---

## 🎨 Design Philosophy

The UI/UX follows a **clean, minimalist, high-aesthetic** design language inspired by:

- **Space & Cosmos**: Color palette derived from planetary themes
- **Medical Professional**: Clean, trustworthy, and professional
- **Modern Web**: Latest design trends and best practices
- **Accessibility**: High contrast, readable fonts, clear hierarchy

### Color Palette

```css
--planetary: #334EAC   /* Deep Blue - Primary actions */
--universe: #7098D1    /* Light Blue - Accents */
--venus: #BAD6EB       /* Pale Blue - Backgrounds */
--meteor: #F7F2EB      /* Off White - Cards */
--galaxy: #081F5C      /* Navy Blue - Text */
--milky-way: #FFF9F0   /* Cream - Background */
--sky: #D0E3FF         /* Sky Blue - Highlights */
```

### Typography

- **Display Font**: Space Grotesk (Headlines, Numbers)
- **Body Font**: Inter (Content, UI elements)
- Both are open-source and available via Google Fonts

---

## 📸 Demo

### Home Page
![Hero Section](https://via.placeholder.com/800x400/334EAC/FFFFFF?text=NeuroScan+AI+Hero)

### Upload & Analysis
![Upload Interface](https://via.placeholder.com/800x400/7098D1/FFFFFF?text=Upload+MRI+Scan)

### Results Visualization
![Results](https://via.placeholder.com/800x400/BAD6EB/334EAC?text=AI+Results+Visualization)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM (for running TensorFlow models)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Step 1: Clone the Repository

```bash
cd MRI-scan-detection-CNN-main
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install web application dependencies
pip install -r requirements-web.txt

# If you also want notebook dependencies:
pip install -r requirements.txt
```

### Step 4: Verify Model Files

Ensure these files exist in the project root:

```
✓ resnet-50-MRI.json          (Classification model architecture)
✓ weights.hdf5                 (Classification model weights)
✓ ResUNet-MRI.json             (Segmentation model architecture)
✓ weights_seg.hdf5             (Segmentation model weights)
✓ utilities.py                 (Custom loss functions)
```

If any model files are missing, you'll need to train them first using `index.ipynb`.

### Dataset Download & Placement

⚠️ **The TCGA dataset folders are not included in this repository due to size limitations.**

Download the dataset and pretrained weights using the provided links and place all extracted `TCGA_*` folders directly in the project root.

📘 For the complete project setup, troubleshooting, and testing instructions, please refer to `SETUP.md`.

**Important:** Do not place them inside an additional parent folder such as `dataset/` or `downloads/`.

Example:

```text
MRI-scan-detection-CNN-main/
├── TCGA_HT_7694_19950404/
├── TCGA_CS_4941_19960909/
├── data.csv
├── data_mask.csv
├── app.py
└── ...
```
### Step 5: Run the Application

```bash
python app.py
```

You should see:

```
============================================================
🧠 NeuroScan AI - Brain Tumor Detection System
============================================================

Initializing application...
Loading classification model...
✓ Classification model loaded successfully
Loading segmentation model...
✓ Segmentation model loaded successfully

✓ All models loaded successfully!

🚀 Starting Flask server...
📍 Access the application at: http://localhost:5000
============================================================
```

### Step 6: Open in Browser

Open your web browser and navigate to:

```
http://localhost:5000
```

---

## 💻 Usage

### 1. Upload MRI Scan

- Click on the upload area or drag & drop an MRI scan image
- Supported formats: JPG, PNG, TIF, TIFF
- Maximum file size: 16MB

### 2. Analyze

- Click the **"Analyze Scan"** button
- Wait for the AI to process (typically 2-3 seconds)
- Watch the animated progress bar

### 3. View Results

The results show:

- ✅ **Detection Status**: Tumor detected or not
- 📊 **Confidence Score**: Model confidence percentage
- 🖼️ **Visualizations**:
  - Original MRI scan
  - AI-generated tumor mask (heatmap)
  - Tumor localization overlay (green highlight)
- 📈 **Tumor Analysis** (if tumor detected):
  - Tumor coverage percentage
  - Number of tumor pixels
  - Detection score

### 4. Analyze More

Click **"Analyze Another Scan"** to upload and analyze additional images.

---

## 🏗️ Project Structure

```
MRI-scan-detection-CNN-main/
│
├── 🌐 Web Application
│   ├── app.py                          # Flask backend server
│   ├── templates/
│   │   └── index.html                  # Main HTML template
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css               # Custom styles
│   │   └── js/
│   │       └── script.js               # Interactive features
│   ├── uploads/                        # Temporary upload directory
│   └── requirements-web.txt            # Web app dependencies
│
├── 🤖 Machine Learning Models
│   ├── resnet-50-MRI.json              # Classification architecture
│   ├── weights.hdf5                    # Classification weights
│   ├── ResUNet-MRI.json                # Segmentation architecture
│   ├── weights_seg.hdf5                # Segmentation weights
│   └── utilities.py                    # Custom loss functions
│
├── 📓 Development
│   ├── index.ipynb                     # Main training notebook
│   ├── data_mask.csv                   # Dataset metadata
│   └── data.csv                        # Image paths
│
├── 📊 Dataset (MRI Scans - Downloaded Separately)
│   ├── TCGA_CS_XXXX_YYYYMMDD/          # Patient directories
│   ├── TCGA_DU_XXXX_YYYYMMDD/          # (110 total)
│   └── ...
│
└── 📄 Documentation
    ├── README.md                       # Original project README
    └── SETUP.md                        # This setup guide
```

---

## 🛠️ Tech Stack

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Custom properties, Grid, Flexbox
- **JavaScript (ES6+)**: Vanilla JS, Fetch API
- **Font Awesome**: Icons
- **Google Fonts**: Inter, Space Grotesk

### Backend
- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **TensorFlow/Keras**: Deep learning models
- **NumPy**: Numerical computations
- **OpenCV**: Image processing
- **Pillow**: Image handling
- **scikit-image**: Advanced image operations

### AI Models
- **ResNet-50**: Transfer learning for classification
- **ResUNet**: Custom U-Net with residual blocks
- **Focal Tversky Loss**: Custom loss function for segmentation

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0.0"
}
```

#### 2. Predict Tumor
```http
POST /api/predict
Content-Type: multipart/form-data
```

**Request Body:**
- `file`: Image file (JPG, PNG, TIF, TIFF)

**Response (No Tumor):**
```json
{
  "has_tumor": false,
  "confidence": 0.978,
  "classification_scores": {
    "no_tumor": 0.978,
    "tumor": 0.022
  },
  "original_image": "data:image/png;base64,..."
}
```

**Response (Tumor Detected):**
```json
{
  "has_tumor": true,
  "confidence": 0.952,
  "classification_scores": {
    "no_tumor": 0.048,
    "tumor": 0.952
  },
  "original_image": "data:image/png;base64,...",
  "segmentation": {
    "mask": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,...",
    "tumor_area_percentage": 12.34,
    "tumor_pixels": 8192
  }
}
```

#### 3. Get Statistics
```http
GET /api/stats
```

**Response:**
```json
{
  "total_patients": 110,
  "total_scans": 3929,
  "model_accuracy": 95.8,
  "segmentation_score": 0.87,
  "average_inference_time": 2.3,
  "model_type": "ResNet-50 + ResUNet"
}
```

---

## 🎓 How It Works

### Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: MRI Scan                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Classification (ResNet-50)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Pre-trained on ImageNet                          │   │
│  │  • Fine-tuned on brain MRI dataset                  │   │
│  │  • Binary output: Tumor / No Tumor                  │   │
│  │  • Confidence score                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │  Tumor Detected? │
                    └─────────────────┘
                         ↙         ↘
                    NO                YES
                     ↓                 ↓
          ┌──────────────────┐  ┌──────────────────────────────┐
          │  Return Results  │  │  Stage 2: Segmentation       │
          │  • No tumor      │  │  (ResUNet)                   │
          │  • Confidence    │  │  ┌────────────────────────┐  │
          └──────────────────┘  │  │ • Encoder-Decoder      │  │
                                │  │ • Residual Blocks      │  │
                                │  │ • Skip Connections     │  │
                                │  │ • Focal Tversky Loss   │  │
                                │  │ • Pixel-level Mask     │  │
                                │  └────────────────────────┘  │
                                └──────────────────────────────┘
                                              ↓
                              ┌────────────────────────────────┐
                              │  OUTPUT: Complete Results      │
                              │  • Classification              │
                              │  • Segmentation Mask           │
                              │  • Tumor Localization          │
                              │  • Quantitative Metrics        │
                              └────────────────────────────────┘
```

---

## 🐛 Troubleshooting

## ⚠️ Important Dataset Setup

The `TCGA_*` folders are **not included in this repository** due to their large size.

Please download the dataset using the links provided above and place all extracted `TCGA_*` folders directly in the project root directory.

❌ Incorrect:

project/
└── dataset/
    └── TCGA_HT_7694_19950404/

✅ Correct:

project/
├── TCGA_HT_7694_19950404/
├── TCGA_CS_4941_19960909/
├── app.py
├── data.csv
└── data_mask.csv

### Dataset Path Errors

If OpenCV cannot load images:
```text
cv2.imread() returns None
(-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'
```python
import os

if not os.path.exists(image_path):
    print(f"Missing file: {image_path}")
```

**Check the following:**

- Ensure all `TCGA_*` folders are placed directly in the project root.
- Ensure the dataset was extracted correctly.
- Ensure CSV paths match the downloaded folder names.
- Ensure the dataset is not inside an additional parent directory after extraction.

### Issue: Models not loading

**Error:**

❌ Failed to load models. Please check that model files exist.


**Solution:**
1. Verify all model files exist in the project root
2. Run the Jupyter notebook (`index.ipynb`) to train/generate models
3. Or download pre-trained models from the project repository

### Issue: Port already in use

**Error:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Find process using port 5000
# Windows:
netstat -ano | findstr :5000

# Linux/Mac:
lsof -i :5000

# Kill the process and restart
```

Or change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Issue: Out of memory

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:**
1. Close other applications
2. Restart the Flask server
3. Reduce batch size (if training)
4. Use a machine with more RAM

### Issue: CORS errors in browser

**Error:**
```
Access to fetch at 'http://localhost:5000/api/predict' has been blocked by CORS policy
```

**Solution:**
- Flask-CORS is already configured in `app.py`
- Clear browser cache and reload
- Check that Flask-CORS is installed: `pip install flask-cors`

---

## 🎨 Customization

### Changing Colors

Edit `static/css/style.css`:

```css
:root {
    --planetary: #YOUR_COLOR;
    --universe: #YOUR_COLOR;
    /* ... */
}
```

### Changing Fonts

Update in `templates/index.html`:

```html
<link href="https://fonts.googleapis.com/css2?family=Your+Font:wght@400;700&display=swap" rel="stylesheet">
```

And in `static/css/style.css`:

```css
:root {
    --font-primary: 'Your Font', sans-serif;
}
```

### Adding New Sections

1. Add HTML section in `templates/index.html`
2. Add styles in `static/css/style.css`
3. Add navigation link in navbar
4. Add scroll spy support in `static/js/script.js`

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Classification Accuracy** | 95.8% |
| **Segmentation Score (Tversky)** | 0.87 |
| **Average Inference Time** | 2.3 seconds |
| **Model Size (Total)** | ~190 MB |
| **Training Dataset** | 3,929 MRI scans |
| **Patients** | 110 unique |

---

## 🔒 Security & Privacy

- ✅ File type validation
- ✅ File size limits (16MB max)
- ✅ Secure file handling
- ✅ No data persistence (uploaded files can be auto-deleted)
- ✅ CORS protection
- ⚠️ **Important**: This is a prototype for educational/research purposes only

---

## 🤝 Contributing

This project was built for a hackathon, but contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the original [README.md](README.md) for details.

---

## 🙏 Acknowledgments

- **Dataset**: TCGA (The Cancer Genome Atlas)
- **ResNet**: Kaiming He et al.
- **U-Net**: Olaf Ronneberger et al.
- **Focal Tversky Loss**: Nabila Abraham & Naimul Mefraz Khan
- **Design Inspiration**: Modern medical imaging applications
- **Color Palette**: Inspired by cosmic themes

---

## 📞 Support

For issues, questions, or suggestions:

- 🐛 [Report a Bug](https://github.com/yourusername/MRI-scan-detection-CNN/issues)
- 💡 [Request a Feature](https://github.com/yourusername/MRI-scan-detection-CNN/issues)
- 📧 Email: your.email@example.com

---

## 🎯 Hackathon Presentation Tips

### Demo Flow
1. **Start with the hero section** - Show the beautiful landing page
2. **Upload a tumor-positive scan** - Demonstrate the drag-and-drop
3. **Show the loading animation** - Highlight the smooth UX
4. **Reveal the results** - Explain the visualizations
5. **Show a negative case** - Demonstrate accuracy
6. **Highlight the architecture** - Explain the two-stage pipeline
7. **Show performance metrics** - Back it up with data

### Key Points to Emphasize
- ✨ Beautiful, professional UI/UX
- 🚀 Real-time AI analysis (< 3 seconds)
- 🎯 High accuracy (95.8%)
- 🔬 Clinical relevance
- 💡 Innovative two-stage approach
- 📱 Fully responsive design

---

<div align="center">

**Built with ❤️ for advancing AI in Healthcare**

⭐ **Star this repo if you found it useful!** ⭐

</div>

---

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Focal Tversky Loss Paper](https://arxiv.org/abs/1810.07842)

---

**Disclaimer**: This application is for research and educational purposes only. It is not intended for clinical diagnosis without proper validation and regulatory approval.