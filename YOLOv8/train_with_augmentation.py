import os

from dotenv import load_dotenv
from ultralytics import YOLO

import wandb
from config import EPOCHS, IMAGE_SIZE, BATCH_SIZE, MODEL_NAME, make_wandb_aug_config
from utils import on_train_epoch_start, on_train_epoch_end, on_train_end


def train_with_augmentation():
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project="msc-thesis",
        name="yolov8n-aug",
        group="yolov8",
        tags=["yolov8n", "n", "aug"],
        config=make_wandb_aug_config(),
        id="y8n-aug"
    )

    try:
        wandb.define_metric("epoch")
        wandb.define_metric("epoch_time_sec", step_metric="epoch")
        wandb.define_metric("epoch_time_min", step_metric="epoch")
        wandb.define_metric("gpu_mem_gb", step_metric="epoch")
        # final metrics (after training) do not require step_metric
    except Exception:
        pass
    model = YOLO(MODEL_NAME, task="detect")

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    model.train(
        data="dataset.yaml",
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        pretrained=False,
        project="msc-thesis",
        name="yolov8n-aug",
        augment=True,
        auto_augment=False,
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

        # disabled default yolo augmentations
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,
    )

    wandb.finish()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    train_with_augmentation()
