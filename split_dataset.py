import os
import shutil
import csv
from sklearn.model_selection import train_test_split

# Base paths
source_base = "dataset"  # Your raw dataset (with images/ and masks/ inside each class)
dest_base = "dataset"        # Destination for train/val splits

# Train/val split ratio
train_ratio = 0.8

# Classes (must match your folder names)
classes = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]

# Assign numeric labels
class_to_label = {cls: idx for idx, cls in enumerate(classes)}

# Make destination folders
for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(os.path.join(dest_base, split, cls, "images"), exist_ok=True)
        os.makedirs(os.path.join(dest_base, split, cls, "masks"), exist_ok=True)

# CSV file writers
train_csv = open("train.csv", "w", newline="")
val_csv = open("val.csv", "w", newline="")
train_writer = csv.writer(train_csv)
val_writer = csv.writer(val_csv)

# Process each class
for cls in classes:
    image_dir = os.path.join(source_base, cls, "images")
    mask_dir = os.path.join(source_base, cls, "masks")

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"[!] Skipping {cls} — missing images/ or masks/ folder.")
        continue

    # List images (png/jpg only)
    images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]

    if len(images) == 0:
        print(f"[!] Skipping {cls} — no image files found.")
        continue

    # Train/val split
    train_imgs, val_imgs = train_test_split(images, train_size=train_ratio, random_state=42)

    # Copy files + write CSV entries
    for img in train_imgs:
        mask = img  # assumes mask has the same filename
        shutil.copy(os.path.join(image_dir, img), os.path.join(dest_base, "train", cls, "images", img))
        shutil.copy(os.path.join(mask_dir, mask), os.path.join(dest_base, "train", cls, "masks", mask))
        train_writer.writerow([
            os.path.join(dest_base, "train", cls, "images", img),
            os.path.join(dest_base, "train", cls, "masks", mask),
            class_to_label[cls]
        ])

    for img in val_imgs:
        mask = img
        shutil.copy(os.path.join(image_dir, img), os.path.join(dest_base, "val", cls, "images", img))
        shutil.copy(os.path.join(mask_dir, mask), os.path.join(dest_base, "val", cls, "masks", mask))
        val_writer.writerow([
            os.path.join(dest_base, "val", cls, "images", img),
            os.path.join(dest_base, "val", cls, "masks", mask),
            class_to_label[cls]
        ])

    print(f"[OK] {cls}: {len(train_imgs)} train, {len(val_imgs)} val")

# Close CSVs
train_csv.close()
val_csv.close()

print("\n✅ Done! train.csv and val.csv created.")
