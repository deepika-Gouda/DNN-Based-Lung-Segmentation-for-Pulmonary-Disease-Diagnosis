import os
import shutil

# Path to the "Viral Pneumonia" folder
base_path = r"dataset\dataset\train\Viral Pneumonia"

# Create 'images' folder if it doesn't exist
images_path = os.path.join(base_path, "images")
os.makedirs(images_path, exist_ok=True)

# Move all PNG, JPG, JPEG files into the 'images' folder
for file in os.listdir(base_path):
    if file.lower().endswith((".png", ".jpg", ".jpeg")) and file != "masks":
        src = os.path.join(base_path, file)
        dst = os.path.join(images_path, file)
        shutil.move(src, dst)
        print(f"Moved {file} -> {dst}")

print("✅ All images moved into 'images' folder!")

