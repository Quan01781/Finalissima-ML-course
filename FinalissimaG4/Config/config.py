class Config:
    PRETRAIN_STAGE = False
    # Dataset
    # DATASET_ROOT = "dataset"
    DATASET_ROOT = "kaggle_dataset" if PRETRAIN_STAGE == True else "dataset"
    NUM_CLASSES = 2 if PRETRAIN_STAGE == True else 4

    # Ablation study
    USE_RESNET_SE = False

    # Label Smoothing
    # LABEL_SMOOTHING = 0.05
    
    # Training
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 20 if PRETRAIN_STAGE == True else 30
    LR = 3e-4
    FREEZE_EPOCHS = 5      #  epochs freeze backbone
    PATIENCE = 5           #  early stopping

    # Model
    MODELS = [
        "efficientnet_b0", # EfficiemtNet
        # "mobilenetv3_small_100", # MobileNet 
        # "shufflenet",
        # "resnetse" 
        ]
    # Default model (fallback)
    MODEL_NAME = "efficientnet_b0"

    # Device
    DEVICE = "cuda"

