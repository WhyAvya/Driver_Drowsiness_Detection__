import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class EyeStateEfficientNet(nn.Module):
    """
    Binary classifier:
    0 -> closed
    1 -> open
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)