import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class TumorDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.imgs = sorted(os.listdir(img_dir))

    def __getitem__(self, idx):
        # Paths to the image and its corresponding label file
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img_name = os.path.splitext(self.imgs[idx])[0]
        label_path = os.path.join(self.label_dir, img_name + ".txt")

        # Load the image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []

        # Load YOLO-format labels and convert to [xmin, ymin, xmax, ymax]
        with open(label_path) as f:
            for line in f:
                cls, x, y, w_box, h_box = map(float, line.strip().split())
                xmin = (x - w_box / 2) * w
                ymin = (y - h_box / 2) * h
                xmax = (x + w_box / 2) * w
                ymax = (y + h_box / 2) * h
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(cls) + 1)  # +1 because background is class 0

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
