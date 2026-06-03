import os
import shutil

# Base paths
BASE_PATH = r"C:\Users\amrut\Downloads\Lung_Segmentation_Website_Project\Lung_Segmentation_Website_Project\dataset\dataset"
OUTPUT_PATH = r"C:\Users\amrut\Downloads\Lung_Segmentation_Website_Project\Lung_Segmentation_Website_Project\dataset"

# Classes in the dataset
CLASSES = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]

def organize_split(split_name):
    """
    Organizes the dataset by copying images and masks into a clean structure.
    Example final path: dataset/train/COVID/images and dataset/train/COVID/masks
    """
    for cls in CLASSES:
        print(f"Processing {cls} in {split_name}...")

        # Source folders
        img_src = os.path.join(BASE_PATH, split_name, cls)              # e.g., dataset/dataset/train/COVID
        mask_src = os.path.join(BASE_PATH, f"{split_name}_masked", cls) # e.g., dataset/dataset/train_masked/COVID

        # Destination folders
        img_dest = os.path.join(OUTPUT_PATH, split_name, cls, "images")
        mask_dest = os.path.join(OUTPUT_PATH, split_name, cls, "masks")

        # Create destination directories
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(mask_dest, exist_ok=True)

        # Copy images
        for file in os.listdir(img_src):
            shutil.copy(os.path.join(img_src, file), os.path.join(img_dest, file))

        # Copy masks
        for file in os.listdir(mask_src):
            shutil.copy(os.path.join(mask_src, file), os.path.join(mask_dest, file))

        print(f"✔ Done organizing {cls} in {split_name}")

# Run for both train and val datasets
organize_split("train")
organize_split("val")

print("✅ Dataset successfully organized into final structure!")
