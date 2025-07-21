import os
from glob import glob
import cv2
import matplotlib.pyplot as plt

def count_images_and_labels(base_path):
    for category in ["Glioma", "Meningioma", "No Tumor", "Pituitary"]:
        image_files = glob(os.path.join(base_path, category, "images", "*.jpg"))
        label_files = glob(os.path.join(base_path, category, "labels", "*.txt"))
        print(f"{category}: {len(image_files)} images, {len(label_files)} labels")

def show_example_images(base_path, category="Glioma", num_images=5):
    image_dir = os.path.join(base_path, category, "images")
    print(f"🔍 Searching for images in: {image_dir}")

    image_paths = glob(os.path.join(image_dir, "*.jpg"))[:num_images]
    if not image_paths:
        print("⚠️ No images found.")
        return

    for path in image_paths:
        print(f"🖼 Displaying: {path}")
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Failed to load image: {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(os.path.basename(path))
        plt.axis('off')
        plt.show()
