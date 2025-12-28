from ultralytics import YOLO
import os
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Validation images (already organized by class folders)
VAL_DIR = os.path.join(PROJECT_ROOT, "data", "neu_surface_defect_cls", "val")

# Trained model weights
MODEL_WEIGHTS = os.path.join(
    PROJECT_ROOT, "results", "dl", "yolo_neu_cls", "weights", "best.pt"
)

# Output folder for visualized predictions
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "dl", "inference")
os.makedirs(OUT_DIR, exist_ok=True)


def put_label(img, text):
    """Draw text at the bottom of the image."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1

    # background box
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = 5
    y = h - 5
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)

    # text
    cv2.putText(img, text, (x, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return img


def main():
    # Load trained model
    model = YOLO(MODEL_WEIGHTS)

    # Loop over class folders in val/
    class_names = [
        d for d in os.listdir(VAL_DIR)
        if os.path.isdir(os.path.join(VAL_DIR, d))
    ]

    num_saved = 0

    for cls in class_names:
        cls_dir = os.path.join(VAL_DIR, cls)
        images = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ]

        # Take up to 5 images per class
        images = images[:5]

        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)

            # YOLOv8 classification inference
            results = model.predict(img_path, imgsz=224, verbose=False)
            res = results[0]

            pred_idx = int(res.probs.top1)
            pred_conf = float(res.probs.top1conf)
            pred_class = res.names[pred_idx]

            # Load image for drawing
            img = cv2.imread(img_path)
            if img is None:
                continue

            label_text = f"Pred: {pred_class} ({pred_conf:.2f}) | True: {cls}"
            vis = put_label(img.copy(), label_text)

            out_filename = f"{cls}__{img_name}"
            out_path = os.path.join(OUT_DIR, out_filename)
            cv2.imwrite(out_path, vis)
            num_saved += 1

    print(f"Saved {num_saved} prediction images to: {OUT_DIR}")


if __name__ == "__main__":
    main()
