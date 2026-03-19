import torch
import torch.nn as nn
from torchvision import models


class EyeStateCNN(nn.Module):
    """
    MobileNetV2 based eye-state classifier.

    Input:  (B, 3, 224, 224)
    Output: (B, 2) logits
    """

    def __init__(self, freeze_backbone=True):
        super(EyeStateCNN, self).__init__()

        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")

        # Freeze early layers
        if freeze_backbone:
            for param in self.backbone.features[:-4].parameters():
                param.requires_grad = False

        # Replace classifier (MATCHES TRAINED MODEL)
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.backbone(x)