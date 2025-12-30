# MSc Thesis - Brain Tumor Detection with YOLO Models

A practical implementation component of the Master's thesis titled **"Analysis and Evaluation of Selected Convolutional Neural Network Models for Brain Tumor Detection in MRI Images"**. This project compares multiple YOLO architecture versions (YOLOv5, YOLOv8, YOLOv10, YOLOv11) in their nano variants for automated brain tumor detection and classification in MRI scans using bounding box detection.

## 📋 Project Overview

This project implements a comprehensive comparison of YOLO-based solutions for detecting and classifying brain tumors in MRI scans across four categories:   
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

The detection approach utilizes **bounding boxes** to localize and classify tumors within MRI images, enabling both detection and spatial localization of abnormalities. 

### Evaluated Models

Four YOLO nano variants were trained and compared under identical conditions on **NVIDIA RTX 3060 6GB**:   

- **YOLOv5n**
- **YOLOv8n**
- **YOLOv10n**
- **YOLOv11n**

## 🗂️ Project Structure

```
msc-thesis/
├── YOLOv8/
│   ├── config.py                       # Configuration parameters
│   ├── data_loader.py                  # Data utilities
│   ├── model.py                        # Model functions
│   ├── utils.py                        # Training callbacks
│   ├── download_dataset.py             # Kaggle downloader
│   ├── prepare_yolo_format.py          # Data preprocessing
│   ├── train_with_augmentation.py      # Training script with augmentation
│   ├── train_without_augmentation.py  # Training script without augmentation
│   ├── detect.py                       # Batch detection
│   └── main.py                         # Single detection
└── data/                               # Dataset directory
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU
- Kaggle API credentials

### Installation

1. Clone the repository:
```bash
git clone https://github.com/chrisitstyle/msc-thesis. git
cd msc-thesis
```

2. Install required packages:
```bash
pip install ultralytics opencv-python matplotlib kaggle python-dotenv wandb torch
```

3. Set up Kaggle API:  
   - Place your `kaggle.json` credentials in `~/.kaggle/`
   - Ensure proper permissions:  `chmod 600 ~/.kaggle/kaggle.json`

### Dataset Setup

1. Download the [MRI Brain Tumor with Bounding Boxes](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes) dataset:

```bash
python YOLOv8/download_dataset.py
```

2. Convert to YOLO format:

```bash
python YOLOv8/prepare_yolo_format.py
```

## 🎯 Training

### Configuration

Adjust training parameters in `config.py`:

```python
EPOCHS = 25
BATCH_SIZE = 32
IMAGE_SIZE = 640
MODEL_NAME = "yolov8n.pt"  # Change for different YOLO versions
```

### Training Execution

```bash
# Train with augmentation
python YOLOv8/train_with_augmentation.py

# Train without augmentation (baseline comparison)
python YOLOv8/train_without_augmentation.py
```

### Augmentation Pipeline

- Geometric:  rotation (±15°), translation (15%), scaling (±20%), shearing (±5°), perspective
- Color: HSV adjustments (hue ±2%, saturation ±50%, brightness ±30%)
- Advanced: horizontal flips (50%), mosaic (30%), mixup (10%)

### Weights & Biases Integration

Optional experiment tracking.  Create `.env` file:

```
WANDB_API_KEY=your_api_key_here
```

## 🔍 Inference

### Single Image

```bash
python YOLOv8/main.py
```

Output saved to `outputs/result.jpg` with bounding boxes drawn around detected tumors. 

### Batch Processing

```bash
python YOLOv8/detect.py
```

Results saved in `detect_results/` directory with visualized bounding boxes. 

## 📊 Model Performance & Comparison

Comprehensive comparison across four YOLO generations (v5, v8, v10, v11) for bounding box detection: 

**Metrics Tracked:**
- Precision
- Recall
- mAP50
- mAP50-95
- F1-score
- GPU memory consumption
- Training time per epoch

**Monitoring Tools:**
- W&B dashboard (comparative analysis)
- Training logs in `runs/detect/train/` (default location, configurable via `project` and `name` parameters in training scripts)
- CSV files with epoch statistics

## 🛠️ Key Features

- **Multi-Architecture Comparison**:  Four YOLO generations evaluated
- **Bounding Box Detection**:  Precise tumor localization with spatial coordinates
- **Augmentation Analysis**:  Compare models trained with and without augmentation
- **Nano Variants**:  Optimized for 6GB VRAM constraints
- **Comprehensive Augmentation**: Geometric, color, and advanced techniques
- **Multi-class Detection**: Four tumor categories simultaneously
- **Experiment Tracking**:  W&B integration for model comparison
- **Resource Monitoring**: GPU memory and training time tracking

## 🔬 Research Applications

**MSc Thesis Title:**
**"Analiza i ocena wybranych modeli sieci splotowych do detekcji guzów mózgu na obrazach MRI"**
*(Analysis and Evaluation of Selected Convolutional Neural Network Models for Brain Tumor Detection in MRI Images)*

**Research Focus:**
- YOLO architecture evolution analysis (v5 → v8 → v10 → v11)
- Bounding box-based tumor detection and localization
- Performance evaluation on resource-constrained hardware
- Impact of augmentation strategies across architectures
- Trade-offs between model complexity, accuracy, and speed
- Practical deployment considerations for medical AI

## 📈 Experimental Results

### Model Comparison - Metrics and Training Computational Cost

**Note:** Models with "-aug" suffix were trained with data augmentation, while base models were trained without augmentation. 

| Model | Precision | Recall | mAP50 | mAP50-95 | F1-score | Avg Training Time (s) | Total Training Time (s) | Avg VRAM Usage (GB) |
|-------|-----------|--------|-------|----------|----------|----------------------|------------------------|---------------------|
| YOLOv5n | 0.945 | 0.922 | 0.958 | 0.739 | 0.933 | 63.19 | 1683.71 | 4.16 |
| YOLOv5n-aug | 0.952 | 0.918 | 0.964 | 0.724 | 0.935 | 65.08 | 1732.79 | 4.16 |
| YOLOv8n | 0.947 | 0.909 | 0.959 | 0.749 | 0.928 | 49.46 | 1340.46 | 4.02 |
| YOLOv8n-aug | 0.923 | 0.919 | 0.962 | 0.738 | 0.921 | 51.72 | 1394.17 | 4.02 |
| YOLOv10n | 0.914 | 0.842 | 0.929 | 0.717 | 0.877 | 66.48 | 1770.32 | 5.63 |
| YOLOv10n-aug | 0.949 | 0.898 | 0.956 | 0.741 | 0.923 | 93.22 | 2459.05 | 5.63 |
| YOLOv11n | 0.947 | 0.916 | 0.959 | 0.748 | 0.931 | 88.26 | 2323.81 | 4.55 |
| YOLOv11n-aug | 0.952 | 0.930 | 0.966 | 0.750 | 0.941 | 90.14 | 2372.34 | 4.55 |

### Analysis

The table presents detection metrics for analyzed YOLO models in two training variants, along with average training time per epoch, total training time, and average VRAM usage.

**Overall Performance:**
- **Best Results**: YOLOv11n with augmentation achieved the highest overall performance across defined metrics:  F1-score of 0.941, mAP50 of 0.966, and mAP50-95 of 0.750
- **Runner-up**: YOLOv5n-aug achieved F1-score of 0.935, demonstrating strong performance despite being an older architecture

**Training Efficiency:**
- **Fastest Training**: YOLOv8n models exhibited the shortest training time (49.46s per epoch, 1340.46s total)
- **Most Resource-Efficient**: YOLOv8n required the least VRAM (4.02 GB)

**Impact of Data Augmentation:**
The analysis reveals that additional processing of training data generally improves model performance, but the effect depends on the specific architecture: 

- **Most Significant Improvement - YOLOv10n**: Augmentation dramatically increased overall precision from 0.914 to 0.949, recall from 0.842 to 0.898, resulting in F1-score improvement from 0.877 to 0.923. This represents a substantial performance gain, significantly reducing the gap between YOLOv10n and other models, demonstrating this architecture's high sensitivity to data augmentation.

- **Consistent Performance - YOLOv5n & YOLOv11n**:  These models showed stable and high metrics in both training variants, with augmentation providing modest but consistent improvements.

- **Architecture-Dependent Results**: The effect of augmentation varies by architecture, with YOLOv10n showing the most dramatic improvement while other models demonstrated more incremental gains.

## 👤 Author

**chrisitstyle**
GitHub:  [@chrisitstyle](https://github.com/chrisitstyle)

## 🙏 Acknowledgments

- Ultralytics YOLO frameworks
- YOLO community
- Kaggle dataset contributors
- Academic supervisors