import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.bench import DetBenchPredict
from effdet.efficientdet import HeadNet
from dataset import MRITumorDataset
from tqdm import tqdm

# ========== PARAMETERS ==========
DATA_DIR     = "data"
OUTPUT_DIR   = "checkpoints"
RESULTS_DIR  = "results"
EPOCHS       = 25
BATCH_SIZE   = 8
LR           = 1e-4
IMG_SIZE     = 512
SCORE_THRESH = 0.5
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

def create_model(num_classes):
    cfg = get_efficientdet_config('tf_efficientdet_d0')
    cfg.num_classes = num_classes
    cfg.image_size  = (IMG_SIZE, IMG_SIZE)
    model = EfficientDet(cfg, pretrained_backbone=True)
    model.class_net = HeadNet(cfg, num_outputs=cfg.num_classes)
    return DetBenchTrain(model, cfg)

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    running_loss = 0.0
    for imgs, targets in pbar:
        imgs_tensor = torch.stack([img.to(device) for img in imgs])
        target = {
            "bbox": [t["boxes"].to(device)  for t in targets],
            "cls":  [t["labels"].to(device) for t in targets]
        }
        out  = model(imgs_tensor, target)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix(train_loss=running_loss / (pbar.n + 1))
    return running_loss / len(loader)

def validate_one_epoch(model, loader, device, epoch):
    model.train()  # keep training branch for loss
    pbar = tqdm(loader, desc=f"  Val Epoch {epoch}")
    running_loss = 0.0
    with torch.no_grad():
        for imgs, targets in pbar:
            imgs_tensor = torch.stack([img.to(device) for img in imgs])
            target = {
                "bbox": [t["boxes"].to(device)  for t in targets],
                "cls":  [t["labels"].to(device) for t in targets]
            }
            out = model(imgs_tensor, target)
            running_loss += out["loss"].item()
            pbar.set_postfix(val_loss=running_loss / (pbar.n + 1))
    return running_loss / len(loader)

def plot_learning_curve(train_losses, val_losses):
    epochs = np.arange(1, len(train_losses)+1)
    plt.figure()
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses,   label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
    plt.close()

def plot_confusion_matrix(cm):
    labels = ["bg"] + MRITumorDataset.CLASSES
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ustawienia osi
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Predicted",
        xlabel="True",
        title="Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # annotacja
    fmt = "d"
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i,j] > thresh else "black"
            )

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close(fig)


def plot_prec_rec_f1(prec, rec, f1):
    labels = ["bg"] + MRITumorDataset.CLASSES
    x = np.arange(len(labels))
    width = 0.2
    plt.figure()
    plt.bar(x - width, prec,  width, label="precision")
    plt.bar(x,         rec,    width, label="recall")
    plt.bar(x + width, f1,     width, label="f1-score")
    plt.xticks(x, labels)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_f1.png"))
    plt.close()

def evaluate_classification(model_trainbench, loader, device):
    effdet_model = model_trainbench.model
    predictor    = DetBenchPredict(effdet_model).to(device)
    predictor.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Eval classification"):
            for img, tgt in zip(imgs, targets):
                inp = img.unsqueeze(0).to(device)
                dets_raw = predictor(inp)  # <- dostajemy Tensor [1, N, 6]
                dets = dets_raw.cpu()[0]    # (N,6)

                gt_lbls  = tgt["labels"]
                true_cls = gt_lbls[0].item() if gt_lbls.numel() > 0 else 0
                y_true.append(true_cls)

                mask = dets[:,4] > SCORE_THRESH
                if mask.any():
                    sub      = dets[mask]
                    top      = sub[sub[:,4].argmax()]
                    pred_cls = int(top[5].item())
                else:
                    pred_cls = 0
                y_pred.append(pred_cls)

    labels = list(range(len(MRITumorDataset.CLASSES)+1))
    cm        = confusion_matrix(y_true, y_pred, labels=labels)
    acc       = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return cm, acc, prec, rec, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor()
    ])

    train_ds = MRITumorDataset(DATA_DIR, split="Train", transforms=transform)
    val_ds   = MRITumorDataset(DATA_DIR, split="Val",   transforms=transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )

    model     = create_model(num_classes=len(MRITumorDataset.CLASSES)+1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        tl = train_one_epoch(model, optimizer, train_loader, device, epoch)
        vl = validate_one_epoch(model, val_loader,   device, epoch)
        train_losses.append(tl)
        val_losses.append(vl)

        print(f"Epoch {epoch:02d}: train_loss={tl:.4f}, val_loss={vl:.4f}")

        ckpt = os.path.join(OUTPUT_DIR, f"effdet_d0_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))

    plot_learning_curve(train_losses, val_losses)
    cm, acc, prec, rec, f1 = evaluate_classification(model, val_loader, device)
    print(f"Val accuracy: {acc:.4f}")
    plot_confusion_matrix(cm)
    plot_prec_rec_f1(prec, rec, f1)

    print("All charts saved in ", RESULTS_DIR)

if __name__ == "__main__":
    main()
