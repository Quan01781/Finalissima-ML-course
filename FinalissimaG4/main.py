import torch
import random
import numpy as np
from Core.dataset import DatasetManager
from Core.model import ModelManager
from Core.trainer import Trainer
from Inference.infer import Inference
from Config.config import Config
from Core.evaluate import Evaluator

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    all_results = []

    # DATA
    dataset = DatasetManager()
    train_loader, val_loader, test_loader = dataset.get_loaders()
    class_names = test_loader.dataset.classes

    for model_name in Config.MODELS:
        Config.MODEL_NAME = model_name
        print(f"\n===== Training {model_name}")
        # MODEL 
        model_mgr = ModelManager()
        model = model_mgr.get_model()
        # TRAINER
        trainer = Trainer(model, train_loader)

        best_f1 = 0.0
        wait = 0

        # SAVE PATH theo stage
        if Config.PRETRAIN_STAGE:
            save_path =  f"pretrained_{Config.MODEL_NAME}.pth"
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
                
                stage = "PRETRAIN" if Config.PRETRAIN_STAGE else "FINETUNE"
                print(f"[{stage}] Saved best model (F1 = {best_f1:.4f})")
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
            evaluator = Evaluator(model, class_names)

            # evaluator = Evaluator(model, class_names)
            # infer = Inference(model, class_names)
            test_f1 = evaluator.compute_f1(test_loader)
            print(f"Test F1-score = {test_f1:.4f}")
            # ROC & MATRIX
            evaluator.plot_confusion_matrix(test_loader)
            evaluator.plot_roc(test_loader)
            # ADD RESULT
            all_results.append({
                "model": Config.MODEL_NAME,
                "f1": test_f1
            })

        else:
            print(f"Pretraining finished. {save_path} saved.")

    # F1 BAR
    Evaluator.plot_f1_bar(all_results)

if __name__ == "__main__":
    main()


