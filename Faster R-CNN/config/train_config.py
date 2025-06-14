import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Base directories
TRAIN_ROOT        = "datasets"
TRAIN_IMG_DIR     = os.path.join(TRAIN_ROOT, "images/train")
TRAIN_LABEL_DIR   = os.path.join(TRAIN_ROOT, "labels/train")

VAL_ROOT          = "data/Val"

# Model / training parameters
NUM_CLASSES       = 5
CLASS_NAMES       = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
BATCH_SIZE        = 2
NUM_EPOCHS        = 2        # adjust as needed
LR                = 0.005
MOMENTUM          = 0.9
SCORE_THRESHOLD   = 0.5

# Output paths
MODEL_SAVE_PATH   = "faster-rcnn_tumor_detector.pth"
METRICS_CSV       = "training_metrics.csv"
METRICS_PNG       = "training_metrics.png"
CM_CSV            = "confusion_matrix.csv"
CM_PNG            = "confusion_matrix.png"
VAL_METRICS_CSV   = "val_metrics.csv"
VAL_METRICS_PNG   = "val_metrics.png"
# ───────────────────────────────────────────────────────────────────────────────
