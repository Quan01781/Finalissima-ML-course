import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    # ROC CURVE (MULTI-CLASS)
    def plot_roc(self, dataloader):
        y_true, _, y_prob = self._collect_outputs(dataloader)
        num_classes = len(self.class_names)

        y_true_bin = label_binarize(
            y_true, classes=list(range(num_classes))
        )

        plt.figure(figsize=(6,5))

        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(
                y_true_bin[:, i], y_prob[:, i]
            )
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr, tpr,
                label=f"{self.class_names[i]} (AUC={roc_auc:.2f})"
            )

        plt.plot([0,1], [0,1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-class ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(self.file_path, "roc.png"), dpi=300, bbox_inches="tight")  
        plt.show()
