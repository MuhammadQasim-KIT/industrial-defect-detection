🏭 Industrial Surface Defect Detection

Classical Computer Vision + YOLOv8 Deep Learning

This project builds an industrial-grade defect classification pipeline combining:

Classical Computer Vision (OpenCV)

YOLOv8 Deep Learning Classification

Early stopping + evaluation metrics

Visual inference outputs

It simulates a real factory surface inspection workflow.

📂 Dataset

Six industrial defect classes:
crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches

Dataset structure:

data/neu_surface_defect/
    train/images/<classes>
    validation/images/<classes>


Converted to YOLO format:

data/neu_surface_defect_cls/
    train/<classes>
    val/<classes>


Dataset preparation:

prepare_neu_cls_from_split.py

🧪 Classical CV Baseline

Implemented rule-based defect highlighting using:

Gaussian blur

Canny edge detection

Otsu thresholding

Morphological filtering

Run:

python classical_cv/baseline_cv.py


Outputs:

results/classical/

🤖 Deep Learning — YOLOv8 Classification
Training Configuration

Model: yolov8n-cls

Input Size: 224×224

Batch: 32

Epochs: 100

Early Stopping Enabled

Optimizer: SGD

Train:

python dl_yolov8_cls/train_yolo_cls.py


Results:

results/dl/yolo_neu_cls/


Includes:

best.pt

results.png

results.csv

confusion_matrix.png

📊 Final Model Performance

✅ Overall Accuracy

Metric	Result
Top-1 Validation Accuracy	100%
Top-5 Validation Accuracy	100%

The model reached perfect validation accuracy with strong stability and excellent generalization.

📉 Training Behavior

Training & validation loss decreased smoothly

No overfitting observed

Early stopping ensured efficient convergence

🧾 Confusion Matrix

Clean diagonal dominance

No misclassifications

Demonstrates strong class separability across all defect types

🖼 Inference Visualization

Inference overlays show:

Predicted label

Confidence score

Ground truth

Run:

python dl_yolov8_cls/infer_yolo_cls.py


Outputs:

results/dl/inference/


Predictions show consistently high confidence (0.97–1.00) with correct classifications.

▶️ How to Run
Environment
conda create -n defectenv python=3.10
conda activate defectenv
pip install ultralytics opencv-python numpy matplotlib torch torchvision

Pipeline
1️⃣ python classical_cv/baseline_cv.py
2️⃣ python prepare_neu_cls_from_split.py
3️⃣ python dl_yolov8_cls/train_yolo_cls.py
4️⃣ python dl_yolov8_cls/infer_yolo_cls.py

🧭 Industrial Perspective

This system can integrate into:

Automated production inspection

Conveyor belt monitoring

Real-time defect classification

Quality assurance pipelines

Key engineering considerations:

Lighting robustness

Edge deployment (ONNX / TensorRT)

Data expansion

Real-time performance tuning

🏁 Summary

This project demonstrates:

✔️ End-to-end industrial defect inspection
✔️ Classical CV vs Deep Learning comparison
✔️ 100% validation accuracy with YOLOv8
✔️ Clear evaluation & visual proof
✔️ Industry-relevant engineering approach
