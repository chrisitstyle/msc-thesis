# config/train_config.py

# paths
DATA_DIR = "./data"
RESULTS_DIR = "./results"

# Dataset
NUM_CLASSES = 4       # glioma, meningioma, no tumor, pituitary
BATCH_SIZE = 8
IMAGE_SIZE = (512, 512)

# Training
NUM_EPOCHS = 6
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9

# Logging & saving
CHECKPOINT_INTERVAL = 5    # Checkpoint every 5 epochs
TENSORBOARD_LOG_DIR = "./results/tensorboard"

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
