import os
import torch
import torchvision
from torch.utils.data import DataLoader
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from dataset import MRITumorDataset
from tqdm import tqdm

# ========== PARAMETERS ==========
DATA_DIR    = "data"
OUTPUT_DIR  = "checkpoints"
EPOCHS      = 25
BATCH_SIZE  = 8
LR          = 1e-4
IMG_SIZE    = 512
# ==========================================

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

        out   = model(imgs_tensor, target)
        loss  = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(train_loss=running_loss / (pbar.n + 1))

    return running_loss / len(loader)

def validate_one_epoch(model, loader, device, epoch):
    # use train() branch but disable grad for validation
    model.train()
    pbar = tqdm(loader, desc=f"  Val Epoch {epoch}")
    running_loss = 0.0

    with torch.no_grad():
        for imgs, targets in pbar:
            imgs_tensor = torch.stack([img.to(device) for img in imgs])
            target = {
                "bbox": [t["boxes"].to(device)  for t in targets],
                "cls":  [t["labels"].to(device) for t in targets]
            }

            out  = model(imgs_tensor, target)
            loss = out["loss"]

            running_loss += loss.item()
            pbar.set_postfix(val_loss=running_loss / (pbar.n + 1))

    return running_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor()
    ])

    # --- Data ---
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

    # --- Model & Optimizer ---
    model     = create_model(num_classes=len(MRITumorDataset.CLASSES) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss   = validate_one_epoch(model, val_loader,   device, epoch)

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # checkpoint after each epoch
        ckpt = os.path.join(OUTPUT_DIR, f"effdet_d0_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)

        # best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"New best model at epoch {epoch}: val_loss={val_loss:.4f}")

    print("Training + validation finished. Checkpoints in ", OUTPUT_DIR)

if __name__ == "__main__":
    main()
