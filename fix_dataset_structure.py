import os
import shutil

# Base path to your train folder
base_path = r"dataset\dataset\train"

# Categories you have
categories = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]

for category in categories:
    cat_path = os.path.join(base_path, category)
    images_folder = os.path.join(cat_path, "images")
    masks_folder = os.path.join(cat_path, "masks")

    # Create subfolders if missing
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    # Move all loose files into images folder
    for file in os.listdir(cat_path):
        full_path = os.path.join(cat_path, file)
        if os.path.isfile(full_path) and file.lower().endswith(('.png', '.jpg')):
            print(f"Moving {file} -> {images_folder}")
            shutil.move(full_path, os.path.join(images_folder, file))
