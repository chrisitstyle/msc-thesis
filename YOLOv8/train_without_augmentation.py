import os

from dotenv import load_dotenv
from ultralytics import YOLO

import wandb
from config import EPOCHS, IMAGE_SIZE, BATCH_SIZE, MODEL_NAME, make_wandb_without_aug_config
from utils import on_train_epoch_start, on_train_epoch_end, on_train_end


# =========================
# main function
# =========================

def train_without_augmentation():
    load_dotenv()

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project="msc-thesis",
        name="yolov8n",
        group="yolov8",
        tags=["yolov8n", "n", "no-aug"],
        config=make_wandb_without_aug_config(),
        id="y8n"
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
        auto_augment=False,
        project="msc-thesis",
        name="yolov8n",
        # additionally and explicitly disable all augmentations
        mosaic=0.0,
        close_mosaic=0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        translate=0.0,
        scale=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,
    )

    wandb.finish()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    train_without_augmentation()
