import os
import sys
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

from config.train_config import (
    DATA_DIR,
    RESULTS_DIR,
    NUM_CLASSES,      # number of tumor classes (4)
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MOMENTUM,
    CHECKPOINT_INTERVAL,
    DEVICE,
)
from model import build_ssd_lite
from dataset import MRIYoloDataset


def collate_fn(batch):
    """Custom collate function to combine samples into a batch for object detection."""
    return tuple(zip(*batch))


def prepare_datasets(root_dir: str, image_size: tuple):
    """Create training and validation datasets with Resize+ToTensor transforms."""
    transforms = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])
    train_ds = MRIYoloDataset(root_dir, image_set="train", transforms=transforms)
    val_ds   = MRIYoloDataset(root_dir, image_set="val",   transforms=transforms)
    return train_ds, val_ds


def plot_confusion(cm: np.ndarray, class_names: list[str], save_path: str):
    """
    Plot a 4x4 confusion matrix with:
      - rows = ground truth labels
      - columns = predicted labels
      - raw counts displayed in each cell
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path}", file=sys.stdout, flush=True)


def train():
    """Main training loop with clear, persistent progress bars printed on stdout."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Initialize the model and optimizer
    model = build_ssd_lite().to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    # Prepare data loaders
    train_ds, val_ds = prepare_datasets(DATA_DIR, IMAGE_SIZE)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Class names for confusion matrix plotting
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    for epoch in range(1, NUM_EPOCHS + 1):
        # 1) Print epoch header
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===", file=sys.stdout, flush=True)

        # 2) Training phase
        model.train()
        total_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc="Training",
            leave=True,
            file=sys.stdout
        )
        for images, targets in train_bar:
            # Move inputs and targets to device
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Forward pass and compute loss
            loss_dict = model(images, targets)
            loss = torch.stack(list(loss_dict.values())).sum()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_bar.close()

        # 3) Print average training loss
        avg_train_loss = total_loss / len(train_loader)
        print(f"Train loss: {avg_train_loss:.4f}", file=sys.stdout, flush=True)

        # 4) Validation phase
        print("Running validation …", file=sys.stdout, flush=True)
        model.eval()
        all_gt = []
        all_pred = []
        ious_list = []

        val_bar = tqdm(
            val_loader,
            desc="Validation",
            leave=True,
            file=sys.stdout
        )
        with torch.no_grad():
            for images, targets in val_bar:
                # Move inputs to device
                images = [img.to(DEVICE) for img in images]
                outputs = model(images)

                # For each image in the batch
                for tgt, out in zip(targets, outputs):
                    gt_boxes = tgt["boxes"].cpu()
                    gt_labels = tgt["labels"].cpu()    # 1..4
                    pred_boxes = out["boxes"].cpu()
                    pred_labels = out["labels"].cpu()  # 1..4

                    # If there are any ground‐truth and predicted boxes
                    if gt_boxes.numel() and pred_boxes.numel():
                        # Compute IoU matrix between GT and predictions
                        iou_matrix = box_iou(gt_boxes, pred_boxes)
                        # For each ground‐truth box, find the best matching prediction
                        for i in range(gt_boxes.shape[0]):
                            max_iou, idx = torch.max(iou_matrix[i], dim=0)
                            if max_iou >= 0.5:
                                # Record for confusion matrix
                                all_gt.append(int(gt_labels[i].item() - 1))
                                all_pred.append(int(pred_labels[idx].item() - 1))
                                # Record IoU value
                                ious_list.append(max_iou.item())
        val_bar.close()

        # 5) Compute and print average IoU
        avg_iou = float(np.mean(ious_list)) if ious_list else 0.0
        print(f"Validation avg IoU: {avg_iou:.4f}", file=sys.stdout, flush=True)

        # 6) Generate and save confusion matrix
        cm = confusion_matrix(all_gt, all_pred, labels=list(range(NUM_CLASSES)))
        plot_confusion(cm, class_names, os.path.join(RESULTS_DIR, "confusion_matrix.png"))

        # 7) Save checkpoint if needed
        if epoch % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(RESULTS_DIR, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}", file=sys.stdout, flush=True)


if __name__ == "__main__":
    train()
