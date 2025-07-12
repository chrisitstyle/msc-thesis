import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from dataset import TumorDataset
from utils import get_model
from config.train_config import (
    TRAIN_IMG_DIR, TRAIN_LABEL_DIR, VAL_ROOT,
    NUM_CLASSES, CLASS_NAMES, BATCH_SIZE, NUM_EPOCHS,
    LR, MOMENTUM, SCORE_THRESHOLD,
    MODEL_SAVE_PATH, METRICS_CSV, METRICS_PNG,
    CM_CSV, CM_PNG, VAL_METRICS_CSV, VAL_METRICS_PNG,
    RESULTS_DIR
)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def collate_fn(batch):
    """Custom collate function for batching."""
    return tuple(zip(*batch))

def gather_val_images(root_dir):
    """Recursively collect all image file paths from the validation directory."""
    imgs = []
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.lower().endswith(VALID_EXTS):
                imgs.append(os.path.join(dp, fn))
    return imgs

def load_image(path, device):
    """Load an image from disk, convert to RGB, transform to tensor, and move to device."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img).to(device)
    return tensor, img

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    # Training DataLoader
    train_ds = TumorDataset(
        img_dir=TRAIN_IMG_DIR,
        label_dir=TRAIN_LABEL_DIR,
        transforms=transforms.ToTensor()
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    tqdm.write(f"Loaded {len(train_ds)} training samples")

    # Model, optimizer and scaler
    model = get_model(NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        momentum=MOMENTUM
    )
    from torch.amp import GradScaler, autocast
    scaler = GradScaler()

    # History metrics
    history = {
        "epoch": [],
        "loss_classifier": [],
        "loss_box_reg": [],
        "loss_objectness": [],
        "loss_rpn_box_reg": [],
        "loss_total": []
    }

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running = {k: 0.0 for k in history if k != "epoch"}

        # Progress bar per epoch
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS}",
            unit="batch",
            leave=True
        )
        for images, targets in pbar:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast("cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update running losses
            for k, v in loss_dict.items():
                running[k] += v.item()
            running["loss_total"] += loss.item()

            # postfix update
            avg_loss = running["loss_total"] / (pbar.n + 1)
            pbar.set_postfix(loss_total=f"{avg_loss:.4f}", refresh=True)

        # epoch end
        final_loss = running["loss_total"] / len(train_loader)
        history["epoch"].append(epoch)
        history["loss_classifier"].append(running["loss_classifier"] / len(train_loader))
        history["loss_box_reg"].append(running["loss_box_reg"] / len(train_loader))
        history["loss_objectness"].append(running["loss_objectness"] / len(train_loader))
        history["loss_rpn_box_reg"].append(running["loss_rpn_box_reg"] / len(train_loader))
        history["loss_total"].append(final_loss)

        tqdm.write(f"-> loss_total epoki {epoch} = {final_loss:.4f}")

        # save training metrics
        df = pd.DataFrame(history)
        df.to_csv(METRICS_CSV, index=False)
        plt.figure()
        for col in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "loss_total"]:
            plt.plot(df["epoch"], df[col], label=col)
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(METRICS_PNG); plt.close()

        torch.cuda.empty_cache()

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    tqdm.write(f"\nModel saved to: {MODEL_SAVE_PATH}")

    # Validation
    tqdm.write("\nRunning validation on Val set...")
    val_paths = gather_val_images(VAL_ROOT)
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for p in tqdm(val_paths, desc="Validation", leave=False):
            tensor, _ = load_image(p, device)
            out = model([tensor])[0]
            labels = out["labels"].cpu().numpy()
            scores = out["scores"].cpu().numpy()

            if len(scores) > 0 and scores.max() >= SCORE_THRESHOLD:
                pred = int(labels[scores.argmax()])
            else:
                pred = CLASS_NAMES.index("No Tumor") + 1
            y_pred.append(pred)

            rel = os.path.relpath(p, VAL_ROOT).split(os.sep)[0]
            y_true.append(CLASS_NAMES.index(rel) + 1 if rel in CLASS_NAMES else 0)

    # Confusion matrix
    labels_all = list(range(1, len(CLASS_NAMES) + 1))
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(CM_CSV)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(CM_PNG); plt.close()

    # Validation metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    val_metrics = {"accuracy": [acc], "precision": [prec], "recall": [rec], "f1_score": [f1]}
    pd.DataFrame(val_metrics).to_csv(VAL_METRICS_CSV, index=False)

    # save validation metrics plot
    plt.figure()
    bars = plt.bar(val_metrics.keys(), [v[0] for v in val_metrics.values()])
    plt.ylim(0, 1); plt.ylabel("Score"); plt.title("Validation Metrics")
    for bar, v in zip(bars, val_metrics.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, f"{v[0]:.2f}", ha="center")
    plt.tight_layout(); plt.savefig(VAL_METRICS_PNG); plt.close()

    tqdm.write(f"\nTraining metrics saved:   {METRICS_CSV}, {METRICS_PNG}")
    tqdm.write(f"Confusion matrix saved:    {CM_CSV}, {CM_PNG}")
    tqdm.write(f"Validation metrics saved:  {VAL_METRICS_CSV}, {VAL_METRICS_PNG}")

if __name__ == "__main__":
    main()
