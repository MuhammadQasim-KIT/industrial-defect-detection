import cv2
import numpy as np
import os
import sys

# Get project root (folder above 'classical_cv')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder with your NEU images
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "neu_surface_defect")

# Output folder
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "classical")
os.makedirs(OUT_DIR, exist_ok=True)


def get_example_image_path():
    """Pick the first .jpg/.png image from the neu_surface_defect folder."""
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")

    candidates = [
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No image files found in {IMG_DIR}. "
            "Make sure your NEU dataset images are inside this folder."
        )

    # Just take the first one
    first_img = candidates[0]
    return os.path.join(IMG_DIR, first_img)


def main():
    img_path = get_example_image_path()
    print(f"Using example image: {img_path}")

    # 1) Read image (as grayscale)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {img_path}")

    # 2) Slight blur to reduce noise
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 3) Edge detection (Canny)
    edges = cv2.Canny(blur, threshold1=50, threshold2=150)

    # 4) Thresholding with Otsu (binary image)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 5) Morphological opening to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6) Dilate edges to make them thicker for visualization
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # 7) Combine thresholded regions and edges
    combined = cv2.bitwise_or(opened, dilated)

    # 8) Save intermediate results
    cv2.imwrite(os.path.join(OUT_DIR, "01_original.png"), img)
    cv2.imwrite(os.path.join(OUT_DIR, "02_blur.png"), blur)
    cv2.imwrite(os.path.join(OUT_DIR, "03_edges.png"), edges)
    cv2.imwrite(os.path.join(OUT_DIR, "04_thresh.png"), thresh)
    cv2.imwrite(os.path.join(OUT_DIR, "05_opened.png"), opened)
    cv2.imwrite(os.path.join(OUT_DIR, "06_dilated.png"), dilated)
    cv2.imwrite(os.path.join(OUT_DIR, "07_combined.png"), combined)

    print("Saved classical CV results in:", OUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
