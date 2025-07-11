import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from effdet import (
    get_efficientdet_config,
    EfficientDet,
    DetBenchTrain,
    DetBenchPredict
)
from effdet.efficientdet import HeadNet
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from dataset import MRITumorDataset

# ========== HYPERPARAMETERS ==========
DATA_DIR    = "data"
OUTPUT_DIR  = "checkpoints"
RESULTS_DIR = "results"
EPOCHS      = 1
BATCH_SIZE  = 8
LR          = 1e-4
IMG_SIZE    = 512
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================================

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

def create_model(num_classes: int) -> DetBenchTrain:
    cfg = get_efficientdet_config('tf_efficientdet_d0')
    cfg.num_classes = num_classes
    cfg.image_size  = (IMG_SIZE, IMG_SIZE)
    model = EfficientDet(cfg, pretrained_backbone=True)
    model.class_net = HeadNet(cfg, num_outputs=cfg.num_classes)
    return DetBenchTrain(model, cfg)

def _run_epoch(model: DetBenchTrain, loader: DataLoader, optimizer, device, is_train: bool) -> float:
    phase = "Train" if is_train else "Val  "
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"{phase} Epoch")

    with torch.set_grad_enabled(is_train):
        for images, targets in pbar:
            imgs = torch.stack([img.to(device) for img in images])
            annots = {
                "bbox": [t["boxes"].to(device)  for t in targets],
                "cls":  [t["labels"].to(device) for t in targets]
            }
            out   = model(imgs, annots)
            loss  = out["loss"]

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            avg = total_loss / (pbar.n + 1)
            pbar.set_postfix(**{f"{phase.lower()}_loss": f"{avg:.4f}"})

    return total_loss / len(loader)

def setup_data_loaders():
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    epochs = np.arange(1, len(train_losses)+1)
    plt.figure()
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses,   label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Learning Curve"); plt.legend()
    plt.savefig(f"{RESULTS_DIR}/loss_curve.png")
    plt.close()

def extract_labels_info(dataset):
    counts = np.zeros(len(dataset.CLASSES), dtype=int)
    xs, ys, ws, hs = [], [], [], []
    for img, target in dataset:
        boxes = target["boxes"].numpy()
        labels= target["labels"].numpy()
        counts += np.bincount(labels, minlength=len(dataset.CLASSES))
        h, w = img.shape[1:]
        xc = ((boxes[:,0] + boxes[:,2]) / 2) / w
        yc = ((boxes[:,1] + boxes[:,3]) / 2) / h
        bw = (boxes[:,2] - boxes[:,0]) / w
        bh = (boxes[:,3] - boxes[:,1]) / h
        xs.extend(xc.tolist()); ys.extend(yc.tolist())
        ws.extend(bw.tolist()); hs.extend(bh.tolist())
    return counts, np.array(xs), np.array(ys), np.array(ws), np.array(hs)

def plot_label_distribution(counts):
    plt.figure()
    plt.bar(MRITumorDataset.CLASSES, counts)
    plt.ylabel("Instances"); plt.title("Label Distribution")
    plt.savefig(f"{RESULTS_DIR}/label_distribution.png")
    plt.close()

def plot_bbox_distributions(xs, ys, ws, hs):
    os.makedirs(f"{RESULTS_DIR}/bboxes", exist_ok=True)
    plt.figure()
    plt.hist2d(xs, ys, bins=50, cmap='Blues')
    plt.xlabel("x_center"); plt.ylabel("y_center")
    plt.title("BBox Centers Heatmap")
    plt.savefig(f"{RESULTS_DIR}/bboxes/centers_heatmap.png")
    plt.close()
    plt.figure()
    plt.hist2d(ws, hs, bins=50, cmap='Blues')
    plt.xlabel("width"); plt.ylabel("height")
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
    x = np.arange(len(MRITumorDataset.CLASSES)); w = 0.25
    plt.figure()
    plt.bar(x - w, prec, w, label="precision")
    plt.bar(x,     rec, w, label="recall")
    plt.bar(x + w, f1,  w, label="f1-score")
    plt.xticks(x, MRITumorDataset.CLASSES)
    plt.ylim(0,1.05); plt.ylabel("Score"); plt.title("Per-Class Metrics"); plt.legend()
    plt.savefig(f"{RESULTS_DIR}/precision_recall_f1.png"); plt.close()

def plot_confidence_curves(y_true, y_pred, confs):
    os.makedirs(f"{RESULTS_DIR}/conf_curves", exist_ok=True)
    thresholds = np.linspace(0,1,101); f1s, precs, recs = [], [], []
    for t in thresholds:
        mask = np.array(confs) >= t
        if mask.sum()==0:
            f1s.append(0); precs.append(0); recs.append(0)
        else:
            yp = np.array(y_pred)[mask]; yt = np.array(y_true)[mask]
            f1s.append(f1_score(yt, yp, average='macro', zero_division=0))
            precs.append(precision_score(yt, yp, average='macro', zero_division=0))
            recs.append(recall_score(yt, yp, average='macro', zero_division=0))
    plt.figure(); plt.plot(thresholds, f1s)
    plt.xlabel("Confidence"); plt.ylabel("F1"); plt.title("F1-Confidence Curve")
    plt.savefig(f"{RESULTS_DIR}/conf_curves/F1_confidence.png"); plt.close()
    plt.figure(); plt.plot(thresholds, precs)
    plt.xlabel("Confidence"); plt.ylabel("Precision"); plt.title("Precision-Confidence Curve")
    plt.savefig(f"{RESULTS_DIR}/conf_curves/Precision_confidence.png"); plt.close()
    plt.figure(); plt.plot(thresholds, recs)
    plt.xlabel("Confidence"); plt.ylabel("Recall");  plt.title("Recall-Confidence Curve")
    plt.savefig(f"{RESULTS_DIR}/conf_curves/Recall_confidence.png"); plt.close()

def evaluate_classification(predictor: DetBenchPredict, loader: DataLoader, device):
    predictor.to(device).eval()
    y_true, y_pred, confs = [], [], []
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Eval classification"):
            imgs = torch.stack([i.to(device) for i in imgs])
            dets = predictor(imgs)             # Tensor [B, K, 6]
            dets = dets.cpu().numpy()
            B = dets.shape[0]
            for b in range(B):
                img_dets = dets[b]            # (K,6)
                if img_dets.shape[0] > 0:
                    j   = img_dets[:,4].argmax()
                    cls = int(img_dets[j,5])
                    cf  = float(img_dets[j,4])
                else:
                    cls, cf = len(MRITumorDataset.CLASSES)-1, 0.0

                # ground truth
                lbls = targets[b]["labels"]
                if lbls.numel() > 0:
                    gt = int(lbls[0].item())
                else:
                    gt = len(MRITumorDataset.CLASSES)-1

                y_true.append(gt)
                y_pred.append(cls)
                confs.append(cf)

    cm   = confusion_matrix(y_true, y_pred, labels=list(range(len(MRITumorDataset.CLASSES))))
    acc  = np.mean(np.array(y_true) == np.array(y_pred))
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
    return cm, acc, prec, rec, f1, y_true, y_pred, confs

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    val_ds, train_loader, val_loader = setup_data_loaders()
    model     = create_model(len(MRITumorDataset.CLASSES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    train_losses, val_losses = [], []
    best_val = float('inf')
    for epoch in range(1, EPOCHS+1):
        tl = _run_epoch(model, train_loader, optimizer, DEVICE, True)
        vl = _run_epoch(model, val_loader,   optimizer, DEVICE, False)
        train_losses.append(tl); val_losses.append(vl)
        print(f"Epoch {epoch:02d}: train_loss={tl:.4f}, val_loss={vl:.4f}")

        ckpt = os.path.join(OUTPUT_DIR, f"effdet_d0_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"→ New best model at epoch {epoch}, val_loss={vl:.4f}")

    # loss and label distribution plots
    plot_learning_curve(train_losses, val_losses)
    counts, xs, ys, ws, hs = extract_labels_info(val_ds)
    plot_label_distribution(counts)
    plot_bbox_distributions(xs, ys, ws, hs)

    # creating a predictor based on the trained effdet.model
    predictor = DetBenchPredict(model.model).to(DEVICE).eval()

    cm, acc, prec, rec, f1, y_t, y_p, confs = evaluate_classification(
        predictor, val_loader, DEVICE
    )
    print(f"Validation accuracy: {acc:.4f}")

    # metrics charts
    plot_confusion_matrix(cm, normalize=False)
    plot_confusion_matrix(cm, normalize=True)
    plot_precision_recall_f1(prec, rec, f1)
    plot_confidence_curves(y_t, y_p, confs)

if __name__ == "__main__":
    main()
