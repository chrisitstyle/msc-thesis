from ultralytics import YOLO
from config import EPOCHS, IMAGE_SIZE, BATCH_SIZE

def train():
    model = YOLO("yolov8n.pt", task="detect")
    model.train(
        data="dataset.yaml",
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        pretrained=False
    )

if __name__ == "__main__":
    train()
