import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import f1_score
from Config.config import Config

class Inference:
    def __init__(self, model, class_names):
        self.model = model
        self.model.eval()  
        self.class_names = class_names
        # self.threshold = 0.3

        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def evaluate(self, test_loader):
        y_true = []  
        y_pred = []  

        # No gradient in test
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)

                outputs = self.model(images)     # forward
                preds = outputs.argmax(dim=1) # class highest prob
                # probs = torch.softmax(outputs, dim=1)
                # preds = (probs[:, 1] > 0.3).long()  # threshold = 0.3

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Macro F1-score 
        f1 = f1_score(y_true, y_pred, average="macro")
        return f1

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(Config.DEVICE)

        with torch.no_grad():
            outputs = self.model(img)
            pred_idx = outputs.argmax(1).item()
            # probs = torch.softmax(outputs, dim=1)
            # pred_idx = int(probs[0, 1] > self.threshold)
            # confidence = probs[0, pred_idx].item()


        return self.class_names[pred_idx]