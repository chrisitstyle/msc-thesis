DATA_DIR = "./data"
TRAIN_DIR = f"{DATA_DIR}/Train"
TEST_DIR = f"{DATA_DIR}/Test"

EPOCHS = 25
BATCH_SIZE = 32
IMAGE_SIZE = 640

MODEL_NAME = "yolov8n.pt"
OUTPUT_MODEL_PATH = "./runs/detect/train/weights/best.pt"


def make_wandb_without_aug_config():
    return dict(
        run=dict(seed=0, deterministic=True, notes="YOLOv8n without augmentations"),
        data=dict(dataset_yaml="dataset.yaml", classes=4),
        model=dict(name=MODEL_NAME, task="detect", pretrained=False),
        train=dict(epochs=EPOCHS, imgsz=IMAGE_SIZE, batch_size=BATCH_SIZE, auto_augment=False, rect=False),
        aug=dict(mosaic=0.0, close_mosaic=0, fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                 translate=0.0, scale=0.0, degrees=0.0, shear=0.0, perspective=0.0, mixup=0.0,
                 cutmix=0.0, copy_paste=0.0, erasing=0.0),
        loss=dict(box=7.5, cls=0.5, dfl=1.5, kobj=1.0),
        val=dict(iou=0.7, max_det=300),
    )


def make_wandb_aug_config():
    return dict(
        run=dict(seed=0, deterministic=True, notes="YOLOv8n with selected augmentations"),
        data=dict(dataset_yaml="dataset.yaml", classes=4),
        model=dict(name=MODEL_NAME, task="detect", pretrained=False),
        train=dict(
            epochs=EPOCHS, imgsz=IMAGE_SIZE, batch_size=BATCH_SIZE,
            auto_augment=False, rect=False
        ),
        # ONLY the augmentations specified in model.train; everything else = 0
        aug=dict(
            # geometry
            degrees=15.0, translate=0.15, scale=0.2, shear=5.0, perspective=0.0002,
            # flips
            fliplr=0.5, flipud=0.0,
            # color
            hsv_h=0.02, hsv_s=0.5, hsv_v=0.3,
            # advanced
            mosaic=0.3, mixup=0.1, close_mosaic=20,
            # disabled
            cutmix=0.0, copy_paste=0.0, erasing=0.0
        ),
        loss=dict(box=7.5, cls=0.5, dfl=1.5, kobj=1.0),
        val=dict(iou=0.7, max_det=300),
    )
