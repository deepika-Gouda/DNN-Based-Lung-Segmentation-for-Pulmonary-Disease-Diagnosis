import pandas as pd
import os

# ------------------------------
# Configuration
# ------------------------------
csv_files = ["dataset/train_labeled.csv", "dataset/val_labeled.csv"]  # CSVs to fix
project_root = os.path.abspath(".")  # Path where train_model.py resides
dataset_folder = os.path.join(project_root, "dataset")  # dataset folder

# ------------------------------
# Helper function to fix paths
# ------------------------------
def fix_path(path):
    # Replace multiple 'dataset' if accidentally repeated
    while "dataset/dataset" in path:
        path = path.replace("dataset/dataset", "dataset")
    # Make absolute path
    full_path = os.path.join(project_root, path)
    return full_path.replace("\\", "/")  # Use forward slashes for consistency

# ------------------------------
# Process each CSV
# ------------------------------
for csv_file in csv_files:
    print(f"\nProcessing CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    missing_images = []
    for idx, row in df.iterrows():
        img_path = row['image_path']
        mask_path = row.get('mask_path', None)  # Optional

        # Fix image path
        fixed_img_path = fix_path(img_path)
        if not os.path.exists(fixed_img_path):
            missing_images.append(img_path)

        # Make path relative to dataset folder
        df.at[idx, 'image_path'] = os.path.relpath(fixed_img_path, dataset_folder)

        # Fix mask path if exists
        if mask_path and isinstance(mask_path, str):
            fixed_mask_path = fix_path(mask_path)
            if not os.path.exists(fixed_mask_path):
                print(f"Warning: Missing mask for {img_path}")
            df.at[idx, 'mask_path'] = os.path.relpath(fixed_mask_path, dataset_folder)

    # Drop rows with missing images
    if missing_images:
        print(f"Found {len(missing_images)} missing images. Removing them from CSV.")
        df = df[~df['image_path'].isin(missing_images)]

    # Save fixed CSV
    fixed_csv_file = csv_file.replace(".csv", "_fixed.csv")
    df.to_csv(fixed_csv_file, index=False)
    print(f"Saved fixed CSV as: {fixed_csv_file}")

print("\n✅ All CSVs processed and fixed for dataset/ folder paths.")
