import timm
import torch
import os
from Config.config import Config
from Core.shufflenet import ShuffleNetV2
from Core.resnetse import ResNet18_SE


class ModelManager:
    def __init__(self):
        # num_classes theo stage
        num_classes = 2 if Config.PRETRAIN_STAGE else Config.NUM_CLASSES

        # ABLATION: RESNET18 + SE
        if Config.USE_RESNET_SE:
            self.model = ResNet18_SE(
                num_classes=num_classes,
                pretrained=True
            )
        # SHUFFLENET
        elif Config.MODEL_NAME == "shufflenet":
            self.model = ShuffleNetV2(
                num_classes=num_classes,
                pretrained=True
            )
        # OTHER MODELS 
        else:
            self.model = timm.create_model(
                Config.MODEL_NAME,
                pretrained=True,   # ImageNet
                num_classes=num_classes
            )

        # LOAD PRETRAINED KAGGLE
        if not Config.PRETRAIN_STAGE:
            ckpt_path = "pretrained_kaggle.pth"
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError("pretrained_kaggle.pth not found")

            print("Loading pretrained Kaggle weights")
            state = torch.load(ckpt_path, map_location=Config.DEVICE)

            # remove classifier 2-class
            filtered = {
                k: v for k, v in state.items()
                if "classifier" not in k
            }

            self.model.load_state_dict(filtered, strict=False)

            # freeze early stages
            self.freeze_early_stages()

        self.model.to(Config.DEVICE)

    # FREEZE 
    def freeze_early_stages(self):
        print("🔒 Freeze early backbone stages")

        for name, param in self.model.named_parameters():
            # RESNET
            if Config.USE_RESNET_SE:
                # Freeze stem + layer1 only
                if name.startswith(("backbone.conv1", "backbone.bn1", "backbone.layer1")):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif Config.MODEL_NAME == "shufflenet":
                    if name.startswith(("backbone.conv1", "backbone.stage2")):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            # EFFICIENTNET / MOBILENET 
            else:
                if any(k in name for k in ["conv_stem", "blocks.0", "blocks.1"]):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def unfreezer(self):
        print("🔓 Unfreeze all layers")
        for p in self.model.parameters():
            p.requires_grad = True

    def get_model(self):
        return self.model
