import os
import csv
import random

# ----------------------------
# CONFIG
# ----------------------------
base_path = "dataset"   # Path to dataset folder containing class subfolders
split_ratio = 0.8       # 80% train, 20% validation
output_train_csv = "dataset/train_labeled.csv"
output_val_csv = "dataset/val_labeled.csv"

pairs = []

# ----------------------------
# Loop through each class folder
# ----------------------------
classes = sorted([cls for cls in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, cls))])
print(f"Found classes: {classes}")

for idx, cls in enumerate(classes):
    cls_path = os.path.join(base_path, cls)
    images_path = os.path.join(cls_path, "images")
    masks_path = os.path.join(cls_path, "masks")

    if not os.path.exists(images_path) or not os.path.exists(masks_path):
        print(f"❌ Skipping {cls} because 'images' or 'masks' folder is missing.")
        continue

    image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = set([f for f in os.listdir(masks_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for img in image_files:
        if img in mask_files:
            # Append tuple: (image_path, mask_path, class_index)
            pairs.append((os.path.join(images_path, img), os.path.join(masks_path, img), idx))
        else:
            print(f"⚠️ No matching mask for: {img} in class {cls}")

# ----------------------------
# Shuffle and split dataset
# ----------------------------
random.shuffle(pairs)
split_index = int(len(pairs) * split_ratio)
train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

# ----------------------------
# Save CSVs
# ----------------------------
def save_csv(filename, data):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path", "label"])
        writer.writerows(data)
    print(f"✅ Saved {filename} ({len(data)} samples)")

save_csv(output_train_csv, train_pairs)
save_csv(output_val_csv, val_pairs)

print(f"Total dataset pairs: {len(pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")
