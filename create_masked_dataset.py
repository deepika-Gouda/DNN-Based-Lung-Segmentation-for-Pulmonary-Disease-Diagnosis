import os
from PIL import Image

# ==============================
# Paths
# ==============================
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TRAIN_MASKED_DIR = os.path.join(BASE_DIR, "train_masked")
VAL_MASKED_DIR = os.path.join(BASE_DIR, "val_masked")

# Ensure output folders exist
os.makedirs(TRAIN_MASKED_DIR, exist_ok=True)
os.makedirs(VAL_MASKED_DIR, exist_ok=True)

# ==============================
# Function to apply mask
# ==============================
def apply_mask(image_path, mask_path, save_path):
    """
    Combines the original image with its segmentation mask
    and saves the result with a red overlay.
    """
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")  # Convert to grayscale

    # Ensure mask matches image size
    if mask.size != image.size:
        print(f"⚠️ Resizing mask for {os.path.basename(image_path)} "
              f"from {mask.size} to {image.size}")
        mask = mask.resize(image.size, Image.BILINEAR)

    # Create red overlay for mask
    red_overlay = Image.new("RGBA", image.size, (255, 0, 0, 120))
    mask_rgba = Image.new("RGBA", image.size)
    mask_rgba.paste(red_overlay, (0, 0), mask)

    # Combine original image with mask overlay
    combined = Image.alpha_composite(image, mask_rgba)

    # Save combined image
    combined.convert("RGB").save(save_path, "PNG")

# ==============================
# Process a single folder
# ==============================
def process_folder(input_dir, output_dir):
    """
    Loops through each class folder and generates masked images.
    """
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Create output class folder
        save_class_path = os.path.join(output_dir, class_name)
        os.makedirs(save_class_path, exist_ok=True)

        print(f"\nProcessing class: {class_name}")
        images_folder = os.path.join(class_path, "images")
        masks_folder = os.path.join(class_path, "masks")

        if not os.path.exists(images_folder) or not os.path.exists(masks_folder):
            print(f"❌ Missing 'images' or 'masks' folder in {class_path}")
            continue

        image_files = sorted(os.listdir(images_folder))
        for image_name in image_files:
            image_path = os.path.join(images_folder, image_name)
            mask_path = os.path.join(masks_folder, image_name)

            if not os.path.exists(mask_path):
                print(f"⚠️ No mask found for {image_name}, skipping...")
                continue

            save_path = os.path.join(save_class_path, image_name)
            apply_mask(image_path, mask_path, save_path)
            print(f"✅ Saved masked image: {save_path}")

# ==============================
# Main script
# ==============================
if __name__ == "__main__":
    print("Processing train set...")
    process_folder(TRAIN_DIR, TRAIN_MASKED_DIR)

    print("\nProcessing validation set...")
    process_folder(VAL_DIR, VAL_MASKED_DIR)

    print("\n🎉 Masked dataset generation complete!")
