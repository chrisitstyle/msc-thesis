import os
import torch
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class MRITumorDataset(Dataset):
    # 0=Glioma, 1=Meningioma, 2=No Tumor, 3=Pituitary
    CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    def __init__(self, root: str, split: str = "Train", transforms=None):
        assert split in ("Train", "Val"), "split must be 'Train' or 'Val'"
        self.transforms = transforms or ToTensor()
        self.img_paths: list[str] = []
        self.label_paths: list[str] = []

        base = os.path.join(root, split)
        for cls_name in self.CLASSES:
            img_dir = os.path.join(base, cls_name, "images")
            lbl_dir = os.path.join(base, cls_name, "labels")
            if not os.path.isdir(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img_p = os.path.join(img_dir, fname)
                lbl_p = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
                if os.path.isfile(lbl_p):
                    self.img_paths.append(img_p)
                    self.label_paths.append(lbl_p)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        # load image in RGB
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        boxes_list: list[list[float]] = []
        labels_list: list[int] = []
        # read YOLO-format label file
        with open(self.label_paths[idx], "r") as f:
            for line in f:
                cls_id, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                boxes_list.append([x1, y1, x2, y2])
                labels_list.append(int(cls_id))

        if boxes_list:
            boxes  = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),    dtype=torch.int64)

        img_tensor = self.transforms(img)
        return img_tensor, {"boxes": boxes, "labels": labels}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
