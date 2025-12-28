import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from Config.config import Config


class Evaluator:
    def __init__(self, model, class_names):
        self.model = model
        self.model.eval()
        self.class_names = class_names
        self.file_path = "C:/Machina Learn/FinalissimaG4"

    # COLLECT PREDICTIONS
    def _collect_outputs(self, dataloader):
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(Config.DEVICE)
                outputs = self.model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        return (
            np.array(y_true),
            np.array(y_pred),
            np.array(y_prob)
        )

    # F1 SCORE for test, visualize
    def compute_f1(self, dataloader):
        y_true, y_pred, _ = self._collect_outputs(dataloader)
        return f1_score(y_true, y_pred, average="macro")

    # F1 BAR CHART
    @staticmethod
    def plot_f1_bar(results):
        models = [r["model"] for r in results]
        f1s = [r["f1"] for r in results]

        plt.figure(figsize=(6,4))
        plt.bar(models, f1s)
        plt.ylabel("F1-macro")
        plt.xlabel("Model")
        plt.title("Model Comparison")

        for i, v in enumerate(f1s):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

        plt.ylim(0, 1)
        plt.savefig(os.path.join("C:/Machina Learn/FinalissimaG4", "f1.png"), dpi=300, bbox_inches="tight")  
        plt.show()

    # CONFUSION MATRIX
    def plot_confusion_matrix(self, dataloader):
        y_true, y_pred, _ = self._collect_outputs(dataloader)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6,5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.file_path, "confusion matrix.png"), dpi=300, bbox_inches="tight")  
        plt.show()

    def _collect_features(self, dataloader):
        features, labels = [], []

        with torch.no_grad():
            for imgs, y in dataloader:
                imgs = imgs.to(Config.DEVICE)

                # FEATURE EXTRACTION
                if hasattr(self.model, "forward_features"):
                    feat = self.model.forward_features(imgs)

                elif hasattr(self.model, "backbone"):
                    feat = self.model.backbone(imgs)

                else:
                    raise ValueError(
                        "Model does not expose forward_features() or backbone"
                    )

                # global pooling
                if feat.dim() == 4:
                    feat = torch.mean(feat, dim=[2, 3])

                features.append(feat.cpu().numpy())
                labels.append(y.numpy())

        return (
            np.concatenate(features, axis=0),
            np.concatenate(labels, axis=0)
        )

    # PCA
    def plot_embedding(self, dataloader):
        X, y = self._collect_features(dataloader)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        plt.figure(figsize=(6, 5))
        for i, cls in enumerate(self.class_names):
            idx = y == i
            plt.scatter(
                X_2d[idx, 0],
                X_2d[idx, 1],
                label=cls,
                alpha=0.6,
                s=20
            )

        plt.legend()
        plt.title("Feature Embedding (PCA)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.savefig(
            os.path.join(self.file_path, "embedding_PCA.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
