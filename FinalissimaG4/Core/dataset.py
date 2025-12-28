from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Config.config import Config

class DatasetManager:
    def __init__(self):
        # Transform for augmentation
        self.train = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize( 
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


        # Val/Test
        self.val_tf = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def get_loaders(self):
        train_ds = datasets.ImageFolder(
            f"{Config.DATASET_ROOT}/train",
            transform=self.train
        )
        val_ds = datasets.ImageFolder(
            f"{Config.DATASET_ROOT}/val",
            transform=self.val_tf
        )
        test_ds = datasets.ImageFolder(
            f"{Config.DATASET_ROOT}/test",
            transform=self.val_tf
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            # in case dataset large
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader
