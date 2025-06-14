import os
import shutil
from glob import glob

# Base directories (paths relative to the project root)
SRC_BASE = os.path.abspath("data")
DST_BASE = os.path.abspath("datasets")

# Class mapping
CLASS_MAP = {
    "Glioma": 0,
    "Meningioma": 1,
    "No Tumor": 2,
    "Pituitary": 3
}

def convert_split(split):
    print(f"🔄 Processing: {split}")
    for cls_name, cls_id in CLASS_MAP.items():
        img_src = os.path.join(SRC_BASE, split, cls_name, "images")
        label_src = os.path.join(SRC_BASE, split, cls_name, "labels")

        img_dst = os.path.join(DST_BASE, "images", split.lower())
        label_dst = os.path.join(DST_BASE, "labels", split.lower())

        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(label_dst, exist_ok=True)

        img_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
            img_paths.extend(glob(os.path.join(img_src, ext)))

        print(f"📂 {cls_name}: found {len(img_paths)} images.")

        for img_path in img_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]  # without extension
            label_path = os.path.join(label_src, img_name + ".txt")

            # preserve the original file extension
            img_ext = os.path.splitext(img_path)[1].lower()
            base_name = f"{cls_name}_{img_name}{img_ext}"

            shutil.copy(img_path, os.path.join(img_dst, base_name))

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    parts[0] = str(cls_id)
                    new_lines.append(" ".join(parts))

                label_target_path = os.path.join(
                    label_dst,
                    base_name.rsplit(".", 1)[0] + ".txt"
                )
                with open(label_target_path, "w") as f:
                    f.write("\n".join(new_lines))
            else:
                print(f"⚠️  Missing label for: {img_path}")

if __name__ == "__main__":
    convert_split("Train")
    convert_split("Val")
    print("✅ Data has been transformed to: datasets/")
