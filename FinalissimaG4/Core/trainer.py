import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Config.config import Config
import numpy as np


class Trainer:
    def __init__(self, model, train_loader):
        self.model = model

        # DETECT HEAD & NUM_CLASSES
        if hasattr(model, "classifier"):
            num_classes = model.classifier.out_features
            head_keywords = ["classifier"]
        elif hasattr(model, "fc") and hasattr(model, "backbone"):
            num_classes = model.fc.out_features
            head_keywords = ["fc"]
        elif hasattr(model, "backbone") and hasattr(model.backbone, "fc"):
            num_classes = model.backbone.fc.out_features
            head_keywords = ["backbone.fc"]
        else:
            raise ValueError("Cannot detect classifier head")

        # SPLIT PARAMS BY NAME
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if any(k in name for k in head_keywords):
                head_params.append(param)
            else:
                backbone_params.append(param)

        # COMPUTE CLASS WEIGHTS
        labels = []
        for _, y in train_loader:
            labels.extend(y.cpu().numpy())

        class_count = np.bincount(labels, minlength=num_classes)

        class_weights = torch.tensor(
            1.0 / (class_count + 1e-6),
            dtype=torch.float
        ).to(Config.DEVICE)

        # LOSS
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # OPTIMIZER
        self.optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": Config.LR * 0.1},
            {"params": head_params, "lr": Config.LR}
        ])

        print(f"Optimizer setup:")
        print(f"  Backbone params: {len(backbone_params)}")
        print(f"  Head params: {len(head_params)}")

    # TRAIN
    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    # EVALUATE for val
    def evaluate(self, val_loader):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(Config.DEVICE)
                outputs = self.model(imgs)
                preds = outputs.argmax(dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        return f1_score(y_true, y_pred, average="macro")
