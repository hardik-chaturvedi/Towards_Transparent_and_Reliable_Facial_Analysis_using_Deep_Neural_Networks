import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
RAW_DATASET = Path("top35")   # original dataset path
OUTPUT_DIR = Path("dataset")  # final processed dataset
SEED = 42
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.2, 0.1

# Fix random seed
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------
# Helper: add salt & pepper noise
# ----------------------------
def add_salt_pepper_noise(image, noise_level=None):
    """
    Adds salt-and-pepper noise to an image.
    noise_level: fraction of pixels to alter (if None, random in [0,0.05])
    """
    if noise_level is None:
        noise_level = random.uniform(0, 0.05)

    noisy = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    num_noisy = int(noise_level * total_pixels)

    # random coords
    coords = [np.random.randint(0, i - 1, num_noisy) for i in image.shape[:2]]

    # half salt, half pepper
    half = num_noisy // 2
    noisy[coords[0][:half], coords[1][:half]] = 255
    noisy[coords[0][half:], coords[1][half:]] = 0

    return noisy

# ----------------------------
# Step 1: Make output dirs
# ----------------------------
for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "val" / "clean").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "val" / "noisy").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "test" / "clean").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "test" / "noisy").mkdir(parents=True, exist_ok=True)

# ----------------------------
# Step 2: Split dataset
# ----------------------------
all_classes = sorted([d.name for d in RAW_DATASET.iterdir() if d.is_dir()])

for cls in all_classes:
    images = list((RAW_DATASET / cls).glob("*.jpg"))
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    # remaining goes to test
    n_test = n_total - n_train - n_val

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train+n_val]
    test_imgs = images[n_train+n_val:]

    # ----------------------------
    # Step 3: Train split (50% clean, 50% noisy)
    # ----------------------------
    train_dir = OUTPUT_DIR / "train" / cls
    train_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(tqdm(train_imgs, desc=f"Train {cls}")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Randomly decide clean/noisy
        if i < len(train_imgs) // 2:
            # save clean
            out_path = train_dir / img_path.name
            cv2.imwrite(str(out_path), img)
        else:
            # save noisy
            noisy_img = add_salt_pepper_noise(img)
            out_path = train_dir / img_path.name.replace(".jpg", "_noisy.jpg")
            cv2.imwrite(str(out_path), noisy_img)

    # ----------------------------
    # Step 4: Val/Test splits (save clean + noisy separately)
    # ----------------------------
    for split_name, split_imgs in [("val", val_imgs), ("test", test_imgs)]:
        for img_path in tqdm(split_imgs, desc=f"{split_name} {cls}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # clean save
            clean_dir = OUTPUT_DIR / split_name / "clean" / cls
            clean_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clean_dir / img_path.name), img)

            # noisy save
            noisy_dir = OUTPUT_DIR / split_name / "noisy" / cls
            noisy_dir.mkdir(parents=True, exist_ok=True)
            noisy_img = add_salt_pepper_noise(img)
            cv2.imwrite(str(noisy_dir / img_path.name.replace(".jpg", "_noisy.jpg")), noisy_img)

print("✅ Dataset preprocessing complete!")
