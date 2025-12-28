import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SRC_TRAIN_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "neu_surface_defect", "train", "images")
SRC_VAL_IMG_DIR   = os.path.join(PROJECT_ROOT, "data", "neu_surface_defect", "validation", "images")

DST_ROOT = os.path.join(PROJECT_ROOT, "data", "neu_surface_defect_cls")
TRAIN_DIR = os.path.join(DST_ROOT, "train")
VAL_DIR   = os.path.join(DST_ROOT, "val")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def copy_split(src_root, dst_root, split_name):
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"{split_name} images folder not found: {src_root}")

    # class folders inside src_root
    class_names = [
        d for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d))
    ]

    total = 0
    for cls in class_names:
        src_cls_dir = os.path.join(src_root, cls)
        dst_cls_dir = os.path.join(dst_root, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        files = [
            f for f in os.listdir(src_cls_dir)
            if f.lower().endswith(IMAGE_EXTS)
        ]

        for fname in files:
            src_path = os.path.join(src_cls_dir, fname)
            dst_path = os.path.join(dst_cls_dir, fname)
            shutil.copy2(src_path, dst_path)
            total += 1

    print(f"[{split_name}] Copied {total} images into {dst_root}")


def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    print("Processing TRAIN split...")
    copy_split(SRC_TRAIN_IMG_DIR, TRAIN_DIR, "train")

    print("Processing VALIDATION split...")
    copy_split(SRC_VAL_IMG_DIR, VAL_DIR, "val")

    print("Done. New dataset root:", DST_ROOT)


if __name__ == "__main__":
    main()
