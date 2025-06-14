import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from utils import get_model

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
MODEL_PATH       = "faster-rcnn_tumor_detector.pth"
INPUT_DIR        = "data/Val"
OUTPUT_DIR       = "predictions/Val"
SCORE_THRESHOLD  = 0.6
NUM_CLASSES      = 5
CLASS_NAMES      = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
# ────────────────────────────────────────────────────────────────────────────────

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def load_image(path):
    """
    Load an image from disk and convert it to a tensor.
    Returns the tensor and the original PIL image.
    """
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img)
    return tensor, img

def draw_predictions(pil_img, boxes, labels, scores):
    """
    Draw bounding boxes and labels on the PIL image for predictions above SCORE_THRESHOLD.
    """
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    for box, lbl, score in zip(boxes, labels, scores):
        if score < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        name = CLASS_NAMES[lbl-1] if (CLASS_NAMES and 1 <= lbl <= len(CLASS_NAMES)) else str(lbl)
        text = f"{name}: {score:.2f}"

        # Compute text width and height using textbbox
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Draw background rectangle and text
        draw.rectangle([x1, y1-th, x1+tw, y1], fill="red")
        draw.text((x1, y1-th), text, fill="white", font=font)

    return pil_img

def gather_images(root_dir):
    """
    Recursively gather all image file paths under root_dir with valid extensions.
    """
    imgs = []
    for dp, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(VALID_EXTS):
                imgs.append(os.path.join(dp, fn))
    return imgs

def main():
    # Select device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Device: {device}")

    # 1. Load the model
    model = get_model(NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # 2. Gather input images
    image_paths = gather_images(INPUT_DIR)
    if not image_paths:
        print(f"❌ No images found in {INPUT_DIR}")
        return
    print(f"🔍 Found {len(image_paths)} images in {INPUT_DIR}")

    # 3. Run predictions and save outputs
    with torch.no_grad():
        for img_path in image_paths:
            tensor, pil = load_image(img_path)
            tensor = tensor.to(device)

            out = model([tensor])[0]
            boxes  = out["boxes"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            scores = out["scores"].cpu().numpy()

            result = draw_predictions(pil.copy(), boxes, labels, scores)

            # Preserve directory structure under OUTPUT_DIR
            rel = os.path.relpath(img_path, INPUT_DIR)
            save_path = os.path.join(OUTPUT_DIR, rel)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            result.save(save_path)
            print(f"✅ Saved: {save_path}")

    print(f"\n🎉 Done! Results are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
