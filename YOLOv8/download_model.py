from ultralytics import YOLO
from config import MODEL_NAME
def download_yolo_model(model_name="yolov8n.pt"): #default model is YOLOv8n
    print(f"🔽 Downloading model: {model_name}...")
    model = YOLO(model_name)  # automatically downloads if not available locally
    print(f"✅ Model '{model_name}' downloaded and ready to use.")

if __name__ == "__main__":
    download_yolo_model(MODEL_NAME)  # you can change to another model, 'yolov8x.pt', 'yolov11n.pt' etc.
