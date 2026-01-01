import torch
import random
import numpy as np
import os
from Core.dataset import DatasetManager
from Core.model import ModelManager
from Core.trainer import Trainer
from Config.config import Config
from Core.evaluate import Evaluator

# FIX RANDOM SEED
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
    
    # PRETRAIN 
    if Config.PRETRAIN_STAGE:
        print("PRETRAIN STAGE")

        model_mgr = ModelManager()
        model = model_mgr.get_model()
        trainer = Trainer(model)

        best_f1 = 0.0
        wait = 0
        for epoch in range(Config.EPOCHS):
            loss = trainer.train(train_loader)
            f1 = trainer.evaluate(val_loader)

            if f1 > best_f1:
                best_f1 = f1
                wait = 0
                torch.save(model.state_dict(), f"pretrained_{Config.MODEL_NAME}.pth")
                print(f"Saved pretrained (F1={best_f1:.4f})")
            else:
                wait += 1
            print(f"[Epoch {epoch+1}/{Config.EPOCHS}] Train Loss={loss:.4f} | F1={f1:.4f}")
            
            # EARLY STOP
            if wait >= Config.PATIENCE:
                print("Early stopping triggered")
                break

        print("PRETRAIN DONE")
        return
    # FINETUNE + COMPARE
    all_results = []
    for model_name in Config.MODELS:
        Config.MODEL_NAME = model_name
        print(f"\n===== {model_name}")
        save_path = f"best_{model_name}.pth"
        # SKIP TRAIN IF EXISTS
        # MODEL 
        model_mgr = ModelManager()
        model = model_mgr.get_model()
        # IF TRAINED
        if os.path.exists(save_path):
            print(f"Found {save_path}, skip training")
            model.load_state_dict(torch.load(save_path))
        else:
            print("Training from scratch (finetune)")
            trainer = Trainer(model)

            best_f1 = 0.0
            wait = 0

            # TRAIN LOOP
            for epoch in range(Config.EPOCHS):

                # UNFREEZE (4-CLASS)
                if epoch == Config.FREEZE_EPOCHS:
                    model_mgr.unfreezer()
                    trainer = Trainer(model)  # reset optimizer
                    print("🔓 Backbone unfrozen")
                # LOSS, F1
                train_loss = trainer.train(train_loader)
                val_f1 = trainer.evaluate(val_loader)

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    wait = 0
                    torch.save(model.state_dict(), save_path)           
                    print(f"Saved best {model_name} (F1 = {best_f1:.4f})")
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

        # EVALUATE
        print("Loading best model for testing")
        model.load_state_dict(torch.load(save_path))
        evaluator = Evaluator(model, class_names)
        test_f1 = evaluator.compute_f1(test_loader)
        print(f"Test F1-score = {test_f1:.4f}")
        # MATRIX
        evaluator.plot_confusion_matrix(test_loader)
        evaluator.plot_embedding(test_loader)
        # ADD RESULT
        all_results.append({
            "model": Config.MODEL_NAME,
            "f1": test_f1
        })


    # F1 BAR
    Evaluator.plot_f1_bar(all_results)
    

if __name__ == "__main__":
    main()