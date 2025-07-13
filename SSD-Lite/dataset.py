import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class MRIYoloDataset(Dataset):

    def __init__(self, root_dir: str, image_set: str, transforms=None):
        """
        root_dir: "./data"
        image_set: "train" or "val"
        """
        self.transforms = transforms
        subset = "Train" if image_set.lower() == "train" else "Val"
        base = os.path.join(root_dir, subset)

        self.img_paths = []
        self.lbl_paths = []
        # collections of images and labels
        for class_name in sorted(os.listdir(base)):
            img_dir = os.path.join(base, class_name, "images")
            lbl_dir = os.path.join(base, class_name, "labels")
            if not os.path.isdir(img_dir):
                continue
            for img_fn in sorted(os.listdir(img_dir)):
                if not img_fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img_path = os.path.join(img_dir, img_fn)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(img_fn)[0] + ".txt")
                if os.path.exists(lbl_path):
                    self.img_paths.append(img_path)
                    self.lbl_paths.append(lbl_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1) Load image
        img = Image.open(self.img_paths[idx]).convert("RGB")
        w, h = img.size

        # 2) Load YOLO labels -> bounding boxes in px
        boxes = []
        labels = []
        with open(self.lbl_paths[idx], "r") as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.strip().split())
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                # torchvision expects labels 1..num_classes, 0=background
                labels.append(int(cls) + 1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        # 3) Transformations
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
