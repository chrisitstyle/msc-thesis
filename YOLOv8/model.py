from ultralytics import YOLO
from config import MODEL_NAME
def load_model(model_path=MODEL_NAME):
    return YOLO(model_path)

def train_model():
    model = YOLO(MODEL_NAME)
    model.train(data="dataset.yaml", epochs=50, imgsz=640, batch=16)

def detect(model, image_path):
    results = model(image_path)
    results[0].show()
    return results
