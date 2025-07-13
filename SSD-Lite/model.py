from torchvision.models.detection import ssd300_vgg16
from torchvision.models import VGG16_Weights
from config.train_config import NUM_CLASSES

def build_ssd_lite(num_classes=NUM_CLASSES):
    model = ssd300_vgg16(
        weights=None,                        # all network from scratch
        weights_backbone=VGG16_Weights.IMAGENET1K_V1,  # only backbone
        num_classes=num_classes + 1            # +1 bg
    )
    return model
