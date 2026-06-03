import os
import shutil

# Path to Lung_Opacity folder
base_path = r"dataset\dataset\train\Lung_Opacity"

# Create 'images' folder if it doesn't exist
images_path = os.path.join(base_path, "images")
os.makedirs(images_path, exist_ok=True)

# Move all image files to 'images' folder
for file in os.listdir(base_path):
    if file.lower().endswith((".png", ".jpg", ".jpeg")) and file != "masks":
        src = os.path.join(base_path, file)
        dst = os.path.join(images_path, file)
        shutil.move(src, dst)
        print(f"Moved {file} -> {dst}")

print("✅ All Lung_Opacity images moved into 'images' folder!")
