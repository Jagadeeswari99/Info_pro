# 🌾 Project 2: Agriculture & Smart Farming - Computer Vision for Crop Disease Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

## 📖 Executive Summary
Global agriculture faces massive challenges as plant diseases threaten food security and crop yields. Traditional manual monitoring of crop health is labor-intensive, prone to human error, and often catches outbreaks too late, leading to devastating crop loss or the indiscriminate overuse of chemical pesticides.

This project develops a **Deep Learning-based image classifier** utilizing Computer Vision. By analyzing images of plant leaves (Bell Pepper, Potato, Tomato), the model detects early signs of disease, enabling timely, targeted interventions and supporting precision agriculture objectives.

---

## 🎯 Business Objectives & KPIs
The strategic vision for this tool is to democratize expert agricultural knowledge, giving local farmers instant, smartphone-accessible diagnostic capabilities.

- **Primary KPI:** **Recall** (Sensitivity). High Recall is critical; failing to identify a diseased leaf (a false negative) allows the infection to spread across the field, which is unacceptable for agricultural deployment. 
- **Secondary KPIs:** Accuracy and Precision.

---

## 🧠 Minimum Viable Product (MVP) Specifications
1. **Image Preprocessing & Augmentation Pipeline:** Standardizes agricultural images (resizing to 224x224, pixel normalization) and handles environmental variability through robust data augmentation (rotations, flips, contrast adjustments).
2. **Convolutional Neural Network (CNN):** A custom CNN architecture built from scratch.
3. **Transfer Learning (ResNet50):** An enterprise-grade architecture achieved by freezing pre-trained ResNet50 base layers and fine-tuning a custom classification head to achieve >90% accuracy.

---

## 🛠 Technology Stack
| Component | Technology | Architectural Rationale |
| :--- | :--- | :--- |
| **Image Processing** | `Python`, `OpenCV`, `PIL` | Industry standard for programmatic image manipulation and resizing. |
| **Deep Learning** | `TensorFlow` / `Keras` | Provides specialized tensor operations and GPU acceleration for deep convolutional layers. |
| **Dataset** | `PlantVillage Dataset` | The gold standard open-source repository containing tens of thousands of labeled crop leaves. |

---

## 📂 Repository Structure
```text
Project_2/
├── data/                   # (Git-Ignored) Place PlantVillage dataset here
├── models/                 # (Git-Ignored) Saved .keras / .h5 models
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── src/                    
│   ├── train.py            # CNN & ResNet50 training pipeline
│   └── inference.py        # Standalone script for raw image prediction
├── .gitignore              # Strict Git bloat prevention
├── requirements.txt        # Exact dependency versions
└── README.md               # You are here
```

---

## 🚀 How to Run Locally

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data
Download the PlantVillage dataset (15 specific classes of Pepper, Potato, and Tomato) and extract it into the `data/PlantVillage/` directory.

### 3. Train the Model
```bash
python src/train.py
```
*This script will load the images, apply augmentation, train the custom CNN, execute Transfer Learning on ResNet50, and output the Confusion Matrix.*

### 4. Run Inference (Prediction)
To predict the disease on a brand new, unseen image:
```bash
python src/inference.py path/to/your/leaf_image.jpg
```
*Output Example:*
```text
========================================
🌾 CROP DISEASE PREDICTION RESULT
========================================
Image Analysed: path/to/your/leaf_image.jpg
Predicted Disease: Tomato___Early_blight
Confidence Score: 98.42%
========================================
```

---

## 📅 2-Week Engineering Roadmap (Completed)
- **Week 1 (Days 1-5):** Image Acquisition, EDA, Preprocessing (OpenCV), and Custom CNN Architecture Baseline Training.
- **Week 2 (Days 6-10):** Transfer Learning (ResNet50 integration), Hyperparameter Optimization, Evaluation (Recall), and Inference Deployment.
