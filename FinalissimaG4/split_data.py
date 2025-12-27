import os
import shutil
import random

SRC = "kaggle_dataset/train"
DST = "kaggle_dataset"

VAL_RATIO = 0.10
TEST_RATIO = 0.20
random.seed(42)

for cls in os.listdir(SRC):
    cls_path = os.path.join(SRC, cls)
    imgs = os.listdir(cls_path)
    random.shuffle(imgs)

    n = len(imgs)
    n_val = int(n * VAL_RATIO)
    n_test = int(n * TEST_RATIO)

    splits = {
        "val": imgs[:n_val],
        "test": imgs[n_val:n_val+n_test],
        "train": imgs[n_val+n_test:]
    }

    for split, files in splits.items():
        dst_dir = os.path.join(DST, split, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for f in files:
            shutil.move(
                os.path.join(cls_path, f),
                os.path.join(dst_dir, f)
            )

print("✅ Kaggle dataset split done")
