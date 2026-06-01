# Contributing to NeuroVision 🧠

Thank you for your interest in contributing to NeuroVision.

NeuroVision is an AI-powered medical imaging project focused on brain tumor classification and segmentation using deep learning techniques. The goal of this project is to build accurate, research-friendly, and efficient models for MRI analysis.

Whether you're improving the codebase, fixing bugs, enhancing documentation, optimizing models, or suggesting new ideas — every contribution is valuable and appreciated.

---

# 📌 About the Project

NeuroVision combines modern deep learning architectures with medical imaging workflows to support:

- Brain tumor classification
- Tumor segmentation
- Attention-based feature learning
- MRI image analysis
- Research experimentation and model development

### Current Models

## Classification Models
- EfficientNetB4
- ResNet-50

## Segmentation Models
- Attention ResUNet v2.0
- CBAM (Convolutional Block Attention Module)
- ASPP (Atrous Spatial Pyramid Pooling)

---

# 🚀 Getting Started

## 1. Fork the Repository

Click the **Fork** button at the top-right corner of the repository page.

Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/neurovision.git
cd neurovision
```

---

## 2. Create a New Branch

Please avoid working directly on the `main` branch.

Create a separate branch for your work:

```bash
git checkout -b feature/your-feature-name
```

Example branch names:

```bash
feature/improve-segmentation
fix/model-loading-error
docs/update-readme
```

---

## 3. Set Up the Environment

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows
```bash
venv\Scripts\activate
```

### Linux/macOS
```bash
source venv/bin/activate
```

Install required dependencies:

```bash
pip install -r requirements-web.txt
```

### Environment variables

Copy the example file and add your credentials (never commit `.env`):

```bash
cp .env.example .env
```

On Windows (Command Prompt): `copy .env.example .env`

Edit `.env` and set at least:

- `FLASK_SECRET_KEY` — a long random string for Flask sessions
- `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` — from [Cloudinary](https://cloudinary.com/)
- `MONGO_URI` (optional) — MongoDB Atlas connection string; omit to use local JSON storage

The app validates required variables at startup and exits with a clear error if any are missing.

---

# 🧪 Development Guidelines

## Code Style

Please write clean, readable, and maintainable code.

### Recommended Practices

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Keep functions modular
- Add comments when necessary
- Avoid unnecessary complexity

---

# 🤖 Model & Research Contributions

If you're contributing model improvements or research experiments, please include:

- Description of the architecture changes
- Training configuration
- Dataset information
- Evaluation metrics
- Performance comparison

### Preferred Metrics

- Accuracy
- Dice Score
- IoU
- Precision
- Recall
- F1-Score

---

# ✅ Commit Message Format

Write clear and meaningful commit messages.

### Good Examples

```bash
feat: add CBAM attention module
fix: resolve dataloader issue
docs: improve setup instructions
refactor: optimize training pipeline
```

### Avoid

```bash
update
fix stuff
changes
```

---

# 📂 Pull Request Process

Before submitting a Pull Request, make sure:

- Your code runs correctly
- Changes are tested locally
- Documentation is updated if needed
- No unnecessary files are included
- Merge conflicts are resolved

---

# 🐞 Reporting Bugs

When reporting issues, please include:

- Operating system
- Python version
- GPU/CUDA information (if applicable)
- Error logs
- Steps to reproduce the issue

Clear bug reports help us solve problems faster.

---

# 💡 Feature Requests

Feature suggestions are always welcome.

Please explain:

- The problem you're trying to solve
- Your proposed solution
- Expected impact or benefits
- Any useful references or research papers

---

# 📖 Documentation Contributions

Documentation improvements are highly appreciated.

You can help by:

- Improving setup instructions
- Fixing typos
- Writing tutorials
- Adding usage examples
- Explaining model architectures

---

# ⚠️ Important Notes

- Do not upload datasets
- Do not commit large model weights/checkpoints
- Keep Pull Requests focused and clean
- Follow the existing project structure
- Write reproducible code whenever possible

---

# 🤝 Code of Conduct

Please maintain a respectful and collaborative environment.

We value:

- Open knowledge sharing
- Constructive discussions
- Research integrity
- Professional behavior

---

# 🌟 Thank You

Thank you for helping improve NeuroVision and supporting AI research in medical imaging.

Every contribution — big or small — truly makes a difference.
