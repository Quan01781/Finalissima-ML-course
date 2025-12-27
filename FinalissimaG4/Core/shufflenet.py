import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
# self.backbone.fc = nn.Linear(in_features, num_classes)

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = shufflenet_v2_x1_0(weights=weights)

        in_features = self.backbone.fc.in_features

        # expose fc 
        self.fc = nn.Linear(in_features, num_classes)
        self.backbone.fc = self.fc

    def forward(self, x):
        return self.backbone(x)
