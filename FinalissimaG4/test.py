import torch
from torchvision import datasets
from Core.model import ModelManager
from Inference.infer import Inference
from Config.config import Config

def main():
    # Load test dataset
    test_dataset = datasets.ImageFolder(
        "dataset/test"
    )
    class_names = test_dataset.classes  # class name

    # Load model
    model_mgr = ModelManager()
    model = model_mgr.get_model()

    # Load trained weight if saved
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(Config.DEVICE)

    # Inference object
    infer = Inference(model, class_names)

    # Predict 
    img_path = "dataset/test/benign/img_001.jpg"
    pred  = infer.predict(img_path)

    print(f"Image: {img_path}")
    print(f"Predicted label: {pred}")

if __name__ == "__main__":
    main()
