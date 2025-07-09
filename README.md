# 🦷 Abnormal Tooth Detection with Dental Enumeration and Diagnosis Using Deep Learning on Panoramic X-rays

> Automating dental anomaly detection with SVM, Faster R-CNN + FPN, and YOLOv8, using the [DENTEX dataset](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)

---

## 📌 Introduction

Manual analysis of dental panoramic X-rays is time-consuming and error-prone. This project aims to automate the detection and diagnosis of dental abnormalities using a combination of classical machine learning (SVM) and deep learning techniques (Faster R-CNN with FPN and YOLOv8).

We trained our models to recognize and classify anomalies such as:
- Dental caries
- Periapical lesions
- Impacted teeth

Our pipeline is built on the **DENTEX dataset**, a hierarchically annotated dataset for dental X-ray diagnostics.

---

## 🗂 Dataset

We used the publicly available dataset:

📦 **DENTEX**: https://huggingface.co/datasets/ibrahimhamamci/DENTEX  
The original dataset had mismatched label distributions across training and test sets. To address this, we:
- Re-split the dataset into train/val/test (7:1.5:1.5)
- Applied data augmentation (rotation and horizontal flipping)
- Increased training set size from 494 to 1976 images

---

## 🔍 Models and Methods

### ✅ SVM + HOG Features
- Traditional baseline using HOG for feature extraction and multi-label SVMs for classification
- Achieved basic detection but limited by anatomical variability

### ✅ Faster R-CNN with FPN
- Backbone: ResNet-50 + FPN for multi-scale detection
- Trained on category 3 (disease), inferred category 1 (quadrant) and 2 (tooth count)
- Outputs: labeled images, number of bounding boxes, quadrant-wise stats

### ✅ YOLOv8
- Real-time object detector using `yolov8n.pt` pre-trained weights
- Converted annotations from JSON to YOLO format
- Trained with `data.yaml` and evaluated using mAP metrics

---

## 📊 Results

| Model               | Precision | Recall | F1 Score |
|--------------------|-----------|--------|----------|
| **Faster R-CNN + FPN** | 0.154     | 0.250  | 0.191    |
| **YOLOv8**              | 0.713     | 0.116  | 0.146    |

> ⚠️ The Faster R-CNN model showed more balanced performance across classes. YOLOv8 excelled in precision but had lower recall on the validation set.

---

## 📦 Project Structure

```bash
├── dataset/                 # Preprocessed DENTEX dataset
├── models/                 # Saved model weights (.pth, .pt)
├── src/                    # Model training, evaluation, and utils
│   ├── train_svm.py
│   ├── train_frcnn.py
│   ├── train_yolo.py
│   └── utils/
├── outputs/                # Inference results and bounding box visualizations
└── README.md
```

---

## 🛠️ Requirements & Setup

```bash
# Clone repo
git clone https://github.com/your_username/dental-xray-detection.git
cd dental-xray-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

Dependencies include:
- PyTorch
- OpenCV
- scikit-learn
- YOLOv8 (via ultralytics)

---

## 🚀 Training & Evaluation

### SVM
```bash
python src/train_svm.py
```

### Faster R-CNN
```bash
python src/train_frcnn.py
```

### YOLOv8
```bash
yolo detect train data=data.yaml model=yolov8n.pt
```

---

## 🧪 Future Work

- Expand dataset with more annotated X-rays
- Improve label consistency across quadrants and diseases
- Explore ensemble models for enhanced accuracy
- Deploy models in clinical decision support systems

---

## ✍️ Authors

- Xu Zhao
- Yuqing Cao
- Linghe Xiao
- Jiahui Fang
- Haonan Jiang

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🔗 Reference

Key dataset: [DENTEX on HuggingFace](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)