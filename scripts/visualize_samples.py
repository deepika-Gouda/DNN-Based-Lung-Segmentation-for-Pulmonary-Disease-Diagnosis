# scripts/visualize_samples.py

import sys
import os

# ----------------------------------------
# Ensure Python can find dataset.py
# ----------------------------------------
# Adds project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
import random

# Import your dataset
from dataset import LungDataset  # <-- dataset.py must be in project root

# ----------------------------------------
# Helper: Unnormalize images for display
# ----------------------------------------
def unnormalize(img_tensor):
    """
    Convert normalized tensor back to displayable image (uint8).
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype('uint8')
    return img

# ----------------------------------------
# Config
# ----------------------------------------
CSV_PATH = "train.csv"  # <-- change to "val.csv" if needed
BATCH_SIZE = 8
IMG_SIZE = 224

# ----------------------------------------
# Load dataset and dataloader
# ----------------------------------------
dataset = LungDataset(CSV_PATH, img_size=IMG_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------------------
# Fetch one batch
# ----------------------------------------
images, masks, labels = next(iter(loader))

# Map numeric labels to class names
CLASS_NAMES = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]

# ----------------------------------------
# Visualize
# ----------------------------------------
for i in range(min(BATCH_SIZE, images.size(0))):
    img = unnormalize(images[i])
    mask = masks[i].squeeze().cpu().numpy()
    label_idx = int(labels[i].item())
    label_name = CLASS_NAMES[label_idx]

    plt.figure(figsize=(6, 3))

    # Left: Image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Class: {label_name} ({label_idx})")
    plt.axis('off')

    # Right: Mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Segmentation Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

print("✅ Visualization complete. Check if masks align with lung regions.")
