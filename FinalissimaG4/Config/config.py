class Config:
    PRETRAIN_STAGE = False
    # Dataset
    # DATASET_ROOT = "dataset"
    DATASET_ROOT = "kaggle_dataset" if PRETRAIN_STAGE == True else "dataset"
    NUM_CLASSES = 2 if PRETRAIN_STAGE == True else 4

    # Ablation study
    USE_RESNET_SE = True

    # Training
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 1e-4
    FREEZE_EPOCHS = 4     #  epochs freeze backbone
    PATIENCE = 5           #  early stopping

    # Model
    MODELS = [
        "efficientnet_b0", # EfficiemtNet
        "mobilenetv3_small_100", # MobileNet 
        "shufflenet",
        "resnetse" 
        ]
    # Default model (fallback)
    MODEL_NAME = ""

    # Device
    DEVICE = "cuda"
