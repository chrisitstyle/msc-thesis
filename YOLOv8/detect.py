import os
import cv2
from glob import glob
from model import load_model

# Paths
IMAGE_DIR = "./data/Test/images"
OUTPUT_DIR = "./detect_results"
MODEL_PATH = "runs/detect/train/weights/best.pt"  # Make sure this exists

# Prepare output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)

# Get all image paths
image_paths = glob(os.path.join(IMAGE_DIR, "*.jpg"))
print(f"🔍 Found {len(image_paths)} images to process.")

# Run detection on each image
for path in image_paths:
    results = model(path)
    img = results[0].plot()

    filename = os.path.basename(path)
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, img)
    print(f"✅ Saved: {save_path}")

print("Detection completed.")