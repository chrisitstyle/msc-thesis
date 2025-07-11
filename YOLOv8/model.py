from ultralytics import YOLO

def load_model(model_path="yolov8n.pt"):
    return YOLO(model_path)

def train_model():
    model = YOLO("yolov8n.pt")
    model.train(data="dataset.yaml", epochs=50, imgsz=640, batch=16)

def detect(model, image_path):
    results = model(image_path)
    results[0].show()
    return results
