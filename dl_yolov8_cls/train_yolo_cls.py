from ultralytics import YOLO
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "neu_surface_defect_cls")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "dl")

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    # 1) Load a small classification model pretrained on ImageNet
    model = YOLO("yolov8n-cls.pt")

    # 2) Train with early stopping and richer logging
    model.train(
        data=DATA_DIR,        # folder containing train/ and val/
        epochs=100,           # max epochs (early stopping will likely stop before)
        imgsz=224,
        batch=32,
        project=RESULTS_DIR,
        name="yolo_neu_cls",
        verbose=True,

        # Early stopping: stop if no improvement in 'patience' epochs
        patience=10,          # you can tune this: 5–15 is common

        # Save settings
        save=True,
        save_period=-1,       # only save best and last
        plots=True,           # save loss curves & metrics plots

        # Optimization
        lr0=0.01,             # initial learning rate (you can adjust later)
        optimizer="SGD",      # or "Adam" / "AdamW"
    )

    print("Training finished. Check results in:", os.path.join(RESULTS_DIR, "yolo_neu_cls"))


if __name__ == "__main__":
    main()
