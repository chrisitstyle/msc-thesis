import os
import shutil
from glob import glob

def flatten_dataset(split, source_split):
    input_dir = f"./data/{source_split}"
    output_img = f"./data/{split}/images"
    output_lbl = f"./data/{split}/labels"

    os.makedirs(output_img, exist_ok=True)
    os.makedirs(output_lbl, exist_ok=True)

    for category in ["Glioma", "Meningioma", "No Tumor", "Pituitary"]:
        cat_path = os.path.join(input_dir, category)
        imgs = glob(os.path.join(cat_path, "images", "*.jpg"))
        lbls = glob(os.path.join(cat_path, "labels", "*.txt"))

        for f in imgs:
            shutil.copy(f, output_img)
        for f in lbls:
            shutil.copy(f, output_lbl)

if __name__ == "__main__":
    # Train from the Train directory
    flatten_dataset("Train", "Train")
    # Test from the Val directory
    flatten_dataset("Test", "Val")
    print("✅ Data prepared in YOLOv8 format.")
