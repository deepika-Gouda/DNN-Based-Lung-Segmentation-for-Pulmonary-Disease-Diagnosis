# dataset.py
import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class LungDataset(Dataset):
    """
    Expects a CSV with columns (flexible):
      - image_path (or image / img)
      - mask_path (or mask)
      - label (or class)

    - If train=True, applies synchronized augmentation to image+mask.
    - Relative paths in the CSV are resolved relative to the CSV file location.
    - Returns: image (tensor, normalized), mask (tensor {0,1}), label (long)
    """
    def __init__(self, csv_file, img_size=224, train=True):
        self.csv_file = csv_file
        self.base_dir = os.path.dirname(os.path.abspath(csv_file)) or os.getcwd()
        self.df = pd.read_csv(csv_file)
        self.img_size = int(img_size)
        self.train = bool(train)

        # Flexible column detection
        cols = [c.lower() for c in self.df.columns]
        def find_col(possible):
            for p in possible:
                if p in cols:
                    return self.df.columns[cols.index(p)]
            return None

        self.image_col = find_col(["image_path", "image", "img", "filepath", "file", "imagefile"])
        self.mask_col  = find_col(["mask_path", "mask", "mask_filepath", "maskfile"])
        self.label_col = find_col(["label", "class", "target"])

        required = (self.image_col, self.mask_col, self.label_col)
        if not all(required):
            raise ValueError(f"CSV must contain columns for image, mask and label. Found columns: {list(self.df.columns)}")

        # ImageNet mean/std for pretrained backbones
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

        # Simple deterministic val transform (no randomness)
        self.val_img_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])
        self.val_mask_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, p):
        if pd.isna(p):
            raise ValueError("Found NaN path in CSV.")
        p = str(p)
        if os.path.isabs(p):
            path = p
        else:
            path = os.path.normpath(os.path.join(self.base_dir, p))
        if not os.path.exists(path):
            print(f"[WARNING] File does not exist: {path}")
        return path

    def _train_transform(self, image, mask):
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        image = TF.resized_crop(image, i, j, h, w, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
        mask  = TF.resized_crop(mask,  i, j, h, w, (self.img_size, self.img_size), interpolation=Image.NEAREST)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle = random.uniform(-15, 15)
        image = image.rotate(angle, resample=Image.BILINEAR)
        mask  = mask.rotate(angle, resample=Image.NEAREST)

        cj = T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)
        image = cj(image)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask

    def _val_transform(self, image, mask):
        image = self.val_img_transform(image)
        mask  = self.val_mask_transform(mask)
        mask  = (mask > 0.5).float()
        return image, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row[self.image_col])
        mask_path = self._resolve_path(row[self.mask_col])
        label = int(row[self.label_col])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            print(f"[ERROR] Failed to open image: {img_path}. Returning black image.")
            image = Image.new("RGB", (self.img_size, self.img_size), 0)

        try:
            mask = Image.open(mask_path).convert("L")
        except Exception:
            print(f"[ERROR] Failed to open mask: {mask_path}. Returning zero mask.")
            mask = Image.new("L", (self.img_size, self.img_size), 0)

        if self.train:
            image, mask = self._train_transform(image, mask)
        else:
            image, mask = self._val_transform(image, mask)

        label = torch.tensor(label, dtype=torch.long)
        return image, mask, label
