import torch
import random
import numpy as np
from Core.dataset import DatasetManager
from Core.model import ModelManager
from Core.trainer import Trainer
from Inference.infer import Inference
from Config.config import Config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)

    # DATA
    dataset = DatasetManager()
    train_loader, val_loader, test_loader = dataset.get_loaders()
    class_names = test_loader.dataset.classes

    # MODEL 
    model_mgr = ModelManager()
    model = model_mgr.get_model()

    trainer = Trainer(model, train_loader)

    best_f1 = 0.0
    wait = 0

    # SAVE PATH theo stage
    if Config.PRETRAIN_STAGE:
        save_path = "pretrained_kaggle.pth"
    else:
        save_path = f"best_{Config.MODEL_NAME}.pth"

    # TRAIN LOOP
    for epoch in range(Config.EPOCHS):

        # UNFREEZE (4-CLASS)
        if (not Config.PRETRAIN_STAGE) and epoch == Config.FREEZE_EPOCHS:
            model_mgr.unfreezer()
            trainer = Trainer(model, train_loader)  # reset optimizer
            print("🔓 Backbone unfrozen")

        train_loss = trainer.train(train_loader)
        val_f1 = trainer.evaluate(val_loader)

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), save_path)

            if Config.PRETRAIN_STAGE:
                print(f"Saved PRETRAIN model (F1 = {best_f1:.4f})")
            else:
                print(f"Saved BEST model (F1 = {best_f1:.4f})")
        else:
            wait += 1

        print(
            f"[Epoch {epoch+1}/{Config.EPOCHS}] "
            f"Train Loss = {train_loss:.4f} | Val F1 = {val_f1:.4f}"
        )

        # EARLY STOP
        if wait >= Config.PATIENCE:
            print("Early stopping triggered")
            break

    # TEST
    if not Config.PRETRAIN_STAGE:
        print("Loading best model for testing")
        model.load_state_dict(torch.load(save_path))

        infer = Inference(model, class_names)
        test_f1 = infer.evaluate(test_loader)
        print(f"Test F1-score = {test_f1:.4f}")

    else:
        print("Pretraining finished. pretrained_kaggle.pth saved.")

if __name__ == "__main__":
    main()


    # Trainer
    # trainer = Trainer(model, train_loader)

    # # Save path 
    # os.makedirs("checkpoints", exist_ok=True)
    # save_path = "checkpoints/best_model.pth"

    # best_f1 = 0.0

    # # Training loop 
    # for epoch in range(Config.EPOCHS):
    #     trainer.train(train_loader)
    #     val_f1 = trainer.evaluate(val_loader)

    #     print(f"[Epoch {epoch+1}/{Config.EPOCHS}] F1 = {val_f1:.4f}")

    #     #  SAVE KHI TỐT HƠN
    #     if val_f1 > best_f1:
    #         best_f1 = val_f1
    #         torch.save(model.state_dict(), save_path)
    #         print(f"Saved BEST model (F1 = {best_f1:.4f})")

    # # Load best model 
    # model.load_state_dict(torch.load(save_path))
    # model.to(Config.DEVICE)

    # # Confusion matrix
    # trainer.plot_confusion_matrix(test_loader, class_names)