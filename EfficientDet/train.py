import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from effdet import (
    get_efficientdet_config,
    EfficientDet,
    DetBenchTrain,
    DetBenchPredict
)
from effdet.efficientdet import HeadNet
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from dataset import MRITumorDataset

# ========== HYPERPARAMETERS ==========
DATA_DIR = "data"
OUTPUT_DIR = "checkpoints"
RESULTS_DIR = "results"
EPOCHS = 25
BATCH_SIZE = 4
LR = 1e-4
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def create_model(num_classes: int) -> DetBenchTrain:
    """
    Build EfficientDet-D0 with a custom head for num_classes.
    """
    cfg = get_efficientdet_config('tf_efficientdet_d0')
    cfg.num_classes = num_classes
    cfg.image_size = (IMG_SIZE, IMG_SIZE)
    model = EfficientDet(cfg, pretrained_backbone=True)
    model.class_net = HeadNet(cfg, num_outputs=cfg.num_classes)
    return DetBenchTrain(model, cfg)


def _run_epoch(
        model: DetBenchTrain,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        is_train: bool,
        epoch: int,
        total_epochs: int
) -> float:
    """
    Run one epoch: if is_train=True, perform training, otherwise validation.
    Displays progress bar with current epoch/total_epochs.
    Returns average loss over the epoch.
    """
    phase = "Train" if is_train else "Val  "
    desc = f"{phase} {epoch}/{total_epochs}"
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=desc)

    with torch.set_grad_enabled(is_train):
        for images, targets in pbar:
            imgs = torch.stack([img.to(device) for img in images])
            annots = {
                "bbox": [t["boxes"].to(device) for t in targets],
                "cls": [t["labels"].to(device) for t in targets]
            }
            out = model(imgs, annots)
            loss = out["loss"]

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix(**{f"{phase.lower()}_loss": f"{avg_loss:.4f}"})

    return total_loss / len(loader)


def setup_data_loaders():
    """
    Prepare training and validation data loaders.
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor()
    ])
    val_ds = MRITumorDataset(DATA_DIR, split="Val", transforms=transform)
    train_ds = MRITumorDataset(DATA_DIR, split="Train", transforms=transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )
    return val_ds, train_loader, val_loader


# --- PLOTTING & METRICS ---

def plot_learning_curve(train_losses, val_losses):
    """
    Plot training and validation loss curves.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o', linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", marker='s', linewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Learning Curve", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/loss_curve.png")
    plt.close()


def plot_metric_curves(accs, maps):
    """
    Plot validation accuracy and mAP over epochs.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    epochs = np.arange(1, len(accs) + 1)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, accs, label="Val Accuracy", marker='o', linewidth=2)
    plt.plot(epochs, maps, label="Val mAP", marker='s', linewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title("Accuracy & mAP over Epochs", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/acc_map_curve.png")
    plt.close()


def extract_labels_info(dataset):
    counts = np.zeros(len(dataset.CLASSES), dtype=int)
    xs, ys, ws, hs = [], [], [], []
    for img, target in dataset:
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()
        counts += np.bincount(labels, minlength=len(dataset.CLASSES))
        h, w = img.shape[1:]
        xc = ((boxes[:, 0] + boxes[:, 2]) / 2) / w
        yc = ((boxes[:, 1] + boxes[:, 3]) / 2) / h
        bw = (boxes[:, 2] - boxes[:, 0]) / w
        bh = (boxes[:, 3] - boxes[:, 1]) / h
        xs.extend(xc.tolist())
        ys.extend(yc.tolist())
        ws.extend(bw.tolist())
        hs.extend(bh.tolist())
    return counts, np.array(xs), np.array(ys), np.array(ws), np.array(hs)


def plot_label_distribution(counts):
    plt.figure()
    plt.bar(MRITumorDataset.CLASSES, counts)
    plt.ylabel("Instances")
    plt.title("Label Distribution")
    plt.savefig(f"{RESULTS_DIR}/label_distribution.png")
    plt.close()


def plot_bbox_distributions(xs, ys, ws, hs):
    os.makedirs(f"{RESULTS_DIR}/bboxes", exist_ok=True)
    plt.figure()
    plt.hist2d(xs, ys, bins=50, cmap='Blues')
    plt.xlabel("x_center")
    plt.ylabel("y_center")
    plt.title("BBox Centers Heatmap")
    plt.savefig(f"{RESULTS_DIR}/bboxes/centers_heatmap.png")
    plt.close()
    plt.figure()
    plt.hist2d(ws, hs, bins=50, cmap='Blues')
    plt.xlabel("width")
    plt.ylabel("height")
    plt.title("BBox Size Heatmap")
    plt.savefig(f"{RESULTS_DIR}/bboxes/size_heatmap.png")
    plt.close()


def plot_confusion_matrix(cmatrix, normalize=False):
    # always float, because we want a common matrix for display
    mat = cmatrix.astype(float)
    if normalize:
        mat = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
        title, fname = "Confusion Matrix Normalized", "confusion_matrix_norm.png"
        fmt = ".2f"
    else:
        title, fname = "Confusion Matrix", "confusion_matrix.png"
        fmt = ".0f"  # instead of d

    plt.figure()
    plt.imshow(mat, cmap='Blues')
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(MRITumorDataset.CLASSES))
    plt.xticks(ticks, MRITumorDataset.CLASSES, rotation=45)
    plt.yticks(ticks, MRITumorDataset.CLASSES)

    thresh = mat.max() / 2
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(
                j, i,
                format(mat[i, j], fmt),
                ha="center", va="center",
                color="white" if mat[i, j] > thresh else "black"
            )

    plt.ylabel("Predicted")
    plt.xlabel("True")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{fname}")
    plt.close()


def plot_precision_recall_f1(prec, rec, f1):
    num = len(prec)
    x = np.arange(num)
    w = 0.25

    plt.figure()
    plt.bar(x - w, prec, w, label="precision")
    plt.bar(x    , rec , w, label="recall")
    plt.bar(x + w, f1  , w, label="f1-score")
    plt.xticks(x, MRITumorDataset.CLASSES[:num], rotation=45)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-Class Metrics")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/precision_recall_f1.png")
    plt.close()


def plot_confidence_curves(y_true, y_pred, confs):
    os.makedirs(f"{RESULTS_DIR}/conf_curves", exist_ok=True)
    thresholds = np.linspace(0, 1, 101)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        mask = np.array(confs) >= t
        if mask.sum() == 0:
            f1s.append(0)
            precs.append(0)
            recs.append(0)
        else:
            yp = np.array(y_pred)[mask]
            yt = np.array(y_true)[mask]
            f1s.append(f1_score(yt, yp, average='macro', zero_division=0))
            precs.append(precision_score(yt, yp, average='macro', zero_division=0))
            recs.append(recall_score(yt, yp, average='macro', zero_division=0))
    plt.figure()
    plt.plot(thresholds, f1s)
    plt.xlabel("Confidence")
    plt.ylabel("F1")
    plt.title("F1-Confidence Curve")
    plt.savefig(f"{RESULTS_DIR}/conf_curves/F1_confidence.png")
    plt.close()
    plt.figure()
    plt.plot(thresholds, precs)
    plt.xlabel("Confidence")
    plt.ylabel("Precision")
    plt.title("Precision-Confidence Curve")
    plt.savefig(f"{RESULTS_DIR}/conf_curves/Precision_confidence.png")
    plt.close()
    plt.figure()
    plt.plot(thresholds, recs)
    plt.xlabel("Confidence")
    plt.ylabel("Recall")
    plt.title("Recall-Confidence Curve")
    plt.savefig(f"{RESULTS_DIR}/conf_curves/Recall_confidence.png")
    plt.close()


def evaluate_classification(predictor: DetBenchPredict, loader: DataLoader, device):
    """
    Run detection, pick top-scoring box per image, compute:
    confusion matrix, accuracy, per-class precision/recall/f1.
    """
    predictor.to(device).eval()
    y_true, y_pred, confs = [], [], []

    no_tumor_idx = MRITumorDataset.CLASSES.index("No Tumor")  # =2

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Eval classification"):
            imgs = torch.stack([i.to(device) for i in imgs])
            dets = predictor(imgs).cpu().numpy()  # [B,K,6]
            for b in range(dets.shape[0]):
                img_dets = dets[b]
                # if detected, take the highest score
                if img_dets.shape[0] > 0:
                    idx = img_dets[:, 4].argmax()
                    cls = int(img_dets[idx, 5])
                    cf = float(img_dets[idx, 4])
                else:
                    # no detection - no tumor
                    cls, cf = no_tumor_idx, 0.0

                # true label
                if targets[b]["labels"].numel() > 0:
                    gt = int(targets[b]["labels"][0].item())
                else:
                    # no boxes in ground-truth - no tumor
                    gt = no_tumor_idx

                y_true.append(gt)
                y_pred.append(cls)
                confs.append(cf)

    labels = list(range(len(MRITumorDataset.CLASSES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    return cm, acc, prec, rec, f1, y_true, y_pred, confs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    val_ds, train_loader, val_loader = setup_data_loaders()
    model = create_model(len(MRITumorDataset.CLASSES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    train_losses, val_losses = [], []
    accuracies, maps = [], []
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # **TRAIN**
        train_loss = _run_epoch(
            model, train_loader, optimizer, DEVICE,
            is_train=True, epoch=epoch, total_epochs=EPOCHS
        )
        # **VALIDATION**
        val_loss = _run_epoch(
            model, val_loader, optimizer, DEVICE,
            is_train=False, epoch=epoch, total_epochs=EPOCHS
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # checkpoints
        ckpt = os.path.join(OUTPUT_DIR, f"effdet_d0_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))

        #  accuracy
        predictor = DetBenchPredict(model.model).to(DEVICE).eval()
        cm, acc, prec, rec, f1, y_true, y_pred, confs = evaluate_classification(
            predictor, val_loader, DEVICE
        )
        accuracies.append(acc)

        #  mAP
        map_metric = MeanAveragePrecision(box_format='xyxy')
        with torch.no_grad():
            for imgs, targets in val_loader:
                batch = torch.stack([i.to(DEVICE) for i in imgs])
                dets = predictor(batch).cpu().numpy()
                preds, gts = [], []
                for b in range(dets.shape[0]):
                    det = dets[b]
                    if det.shape[0] and targets[b]['boxes'].size(0):
                        boxes = torch.tensor(det[:, :4])
                        scores = torch.tensor(det[:, 4])
                        labels = torch.tensor(det[:, 5], dtype=torch.int64)
                    else:
                        boxes = torch.zeros((0, 4))
                        scores = torch.zeros((0,))
                        labels = torch.zeros((0,), dtype=torch.int64)
                    preds.append({'boxes': boxes, 'scores': scores, 'labels': labels})
                    gts.append({'boxes': targets[b]['boxes'], 'labels': targets[b]['labels']})
                map_metric.update(preds, gts)
        map_score = map_metric.compute()['map'].item()
        maps.append(map_score)

        print(
            f"Epoch {epoch:02d}: lr={current_lr:.2e}, "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"acc={acc:.4f}, mAP={map_score:.4f}"
        )

    # plot curves
    plot_learning_curve(train_losses, val_losses)
    plot_metric_curves(accuracies, maps)

    # Final detailed plots
    counts, xs, ys, ws, hs = extract_labels_info(val_ds)
    plot_label_distribution(counts)
    plot_bbox_distributions(xs, ys, ws, hs)
    plot_confusion_matrix(cm, normalize=False)
    plot_confusion_matrix(cm, normalize=True)
    plot_precision_recall_f1(prec, rec, f1)
    plot_confidence_curves(y_true, y_pred, confs)


if __name__ == "__main__":
    main()
