import os
import cv2
import torch
import torchvision
import numpy as np
from effdet import get_efficientdet_config, EfficientDet
from effdet.bench import DetBenchPredict
from effdet.efficientdet import HeadNet
from dataset import MRITumorDataset

# ========== SETTINGS ==========
IMG_SIZE     = 512
MODEL_PATH   = "checkpoints/best_model.pth"
IMG_DIR      = "imgToDetect"
OUT_DIR      = "detectedImgs"
SCORE_THRESH = 0.5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Build EfficientDet & HeadNet
cfg = get_efficientdet_config('tf_efficientdet_d0')
cfg.num_classes = len(MRITumorDataset.CLASSES) + 1
cfg.image_size  = (IMG_SIZE, IMG_SIZE)
effdet = EfficientDet(cfg, pretrained_backbone=False)
effdet.class_net = HeadNet(cfg, num_outputs=cfg.num_classes)

# 2) Load raw state_dict and strip "model." prefix
raw_state = torch.load(MODEL_PATH, map_location=DEVICE)
clean_state = {}
for k, v in raw_state.items():
    if k.startswith("model."):
        clean_state[k[len("model."):]] = v
    else:
        clean_state[k] = v

effdet.load_state_dict(clean_state, strict=False)

# 3) Wrap in predictor (only model, no cfg)
predictor = DetBenchPredict(effdet).to(DEVICE)
predictor.eval()

# 4) Preprocessing
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor()
])

classnames = ["bg"] + MRITumorDataset.CLASSES

# 5) Inference loop
for fname in os.listdir(IMG_DIR):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMG_DIR, fname)
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        continue

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        dets = predictor(inp)[0].cpu().numpy()  # (N,6)

    H, W = img_bgr.shape[:2]
    scale_x = W / IMG_SIZE
    scale_y = H / IMG_SIZE

    for x1, y1, x2, y2, score, cls in dets:
        if score < SCORE_THRESH:
            continue
        x1o = int(x1 * scale_x)
        y1o = int(y1 * scale_y)
        x2o = int(x2 * scale_x)
        y2o = int(y2 * scale_y)

        label = f"{classnames[int(cls)]}:{score:.2f}"
        cv2.rectangle(img_bgr, (x1o, y1o), (x2o, y2o), (0,255,0), 2)
        cv2.putText(
            img_bgr, label, (x1o, y1o - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2,
            lineType=cv2.LINE_AA
        )

    cv2.imwrite(os.path.join(OUT_DIR, fname), img_bgr)

print(f"✅ Detection complete. Results saved to '{OUT_DIR}/'")
