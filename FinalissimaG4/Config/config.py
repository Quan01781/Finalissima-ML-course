class Config:
    # Dataset
    # DATASET_ROOT = "dataset"
    DATASET_ROOT = "kaggle_dataset"
    # pretrained_kaggle.pth
    NUM_CLASSES = 4
    PRETRAIN_STAGE = True

    # Ablation study
    USE_RESNET_SE = False

    # Training
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 1e-4
    FREEZE_EPOCHS = 4      #  epochs freeze backbone
    PATIENCE = 5           #  early stopping

    # Model
    MODELS = [
        "efficientnet_b0", # EfficiemtNet
        "mobilenetv3_small_100", # MobileNet 
        "shufflenet"  
        ]
    # Default model (fallback)
    MODEL_NAME = MODELS[2]

    # Device
    DEVICE = "cuda"
