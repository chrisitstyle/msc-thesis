from ultralytics import YOLO
from config import EPOCHS, IMAGE_SIZE, BATCH_SIZE, MODEL_NAME


def train_with_augmentation():
    model = YOLO(MODEL_NAME, task="detect")
    model.train(
        data="dataset.yaml",
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        pretrained=False,
        augment=True,
        auto_augment=False,  # Disable AutoAugment
        # Geometric transformations
        degrees=15.0,  # Rotation ±15°
        translate=0.15,  # Translation 15%
        scale=0.2,  # Scaling ±20%
        shear=5.0,  # Shearing ±5°
        perspective=0.0002,  # Perspective

        # Flips
        fliplr=0.5,  # Horizontal flip 50%
        flipud=0.0,  # No vertical flip

        # Color transformations
        hsv_h=0.02,  # Hue ±2%
        hsv_s=0.5,  # Saturation ±50%
        hsv_v=0.3,  # Brightness ±30%

        # Advanced techniques
        mosaic=0.3,  # Mosaic 30%
        mixup=0.1,  # Mixup 10%

        # Mosaic closure towards the end of training
        close_mosaic=20,  # Disable mosaic 20 epochs before the end
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    train_with_augmentation()
