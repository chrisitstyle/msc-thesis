import os
import cv2
from config import TRAIN_DIR
from data_loader import count_images_and_labels, show_example_images
from model import load_model, detect

# 1. Data exploration
count_images_and_labels(TRAIN_DIR)
show_example_images(TRAIN_DIR, category="Glioma")

# 2. Load model and run detection
# model = load_model("yolov8n.pt")
model = load_model("runs/detect/train/weights/best.pt")
example_image = "./data/Test/images/gg (9).jpg"  # Set a valid image path
results = detect(model, example_image)

# 3. Save the result to a file
os.makedirs("outputs", exist_ok=True)
img = results[0].plot()
cv2.imwrite("outputs/result.jpg", img)
print("✅ Result saved to outputs/result.jpg")
