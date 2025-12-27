import torch
import torch.nn as nn
import timm
from Core.attention import SEModule


# RESNET18 + SE
class ResNet18_SE(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()

        self.backbone = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            num_classes=0,     # reset classifier
            # global_pool=""
        )

        # SE equiped to layers
        self.se3 = SEModule(256, reduction=16)
        self.se4 = SEModule(512, reduction=16)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        # Layers (no SE)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        # Deep layers + SE
        x = self.backbone.layer3(x)
        x = self.se3(x)

        x = self.backbone.layer4(x)
        x = self.se4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
