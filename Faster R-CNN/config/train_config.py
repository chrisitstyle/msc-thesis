import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Base directories
TRAIN_ROOT        = "datasets"
TRAIN_IMG_DIR     = os.path.join(TRAIN_ROOT, "images/train")
TRAIN_LABEL_DIR   = os.path.join(TRAIN_ROOT, "labels/train")

VAL_ROOT          = "data/Val"

# Model / training parameters
#Faster R-CNN needs num_classes = number of objects + 1 for background,
# but for CM and names we keep only the true 4 classes)
NUM_CLASSES       = 5
CLASS_NAMES       = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

BATCH_SIZE        = 2
NUM_EPOCHS        = 25
LR                = 0.005
MOMENTUM          = 0.9
SCORE_THRESHOLD   = 0.5

# Output directory
RESULTS_DIR       = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Output paths (all in results/)
MODEL_SAVE_PATH   = os.path.join(RESULTS_DIR, "faster-rcnn_tumor_detector.pth")
METRICS_CSV       = os.path.join(RESULTS_DIR, "training_metrics.csv")
METRICS_PNG       = os.path.join(RESULTS_DIR, "training_metrics.png")
CM_CSV            = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
CM_PNG            = os.path.join(RESULTS_DIR, "confusion_matrix.png")
VAL_METRICS_CSV   = os.path.join(RESULTS_DIR, "val_metrics.csv")
VAL_METRICS_PNG   = os.path.join(RESULTS_DIR, "val_metrics.png")
# ───────────────────────────────────────────────────────────────────────────────
