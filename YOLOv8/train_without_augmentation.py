from ultralytics import YOLO

from config import EPOCHS, IMAGE_SIZE, BATCH_SIZE, MODEL_NAME


def train_without_augmentation():
    model = YOLO(MODEL_NAME, task="detect")
    model.train(
        data="dataset.yaml",
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        pretrained=False
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    train_without_augmentation()