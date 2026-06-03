# train_model.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from project_model.resnet_seg import ResNetSegmentationClassification
from dataset import LungDataset

def train_one_epoch(model, dataloader, optimizer, criterion_seg, criterion_cls, device, alpha=0.5, beta=1.0):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    total_seg, total_cls = 0, 0

    for images, masks, labels in tqdm(dataloader, desc="Train", leave=False):
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        masks = masks.float() / 255.0 if masks.max() > 1 else masks.float()
        labels = labels.long()

        output = model(images)
        seg_output, cls_output = output["seg_mask"], output["logits"]

        loss_seg = criterion_seg(seg_output, masks)
        loss_cls = criterion_cls(cls_output, labels)
        loss = alpha * loss_seg + beta * loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_seg += loss_seg.item() * bs
        total_cls += loss_cls.item() * bs
        total_correct += (torch.argmax(cls_output, dim=1) == labels).sum().item()
        total_samples += bs

    return total_loss/total_samples, total_correct/total_samples, total_seg/total_samples, total_cls/total_samples

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion_seg, criterion_cls, device, alpha=0.5, beta=1.0):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    total_seg, total_cls = 0, 0

    for images, masks, labels in tqdm(dataloader, desc="Val", leave=False):
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        masks = masks.float() / 255.0 if masks.max() > 1 else masks.float()
        labels = labels.long()

        output = model(images)
        seg_output, cls_output = output["seg_mask"], output["logits"]

        loss_seg = criterion_seg(seg_output, masks)
        loss_cls = criterion_cls(cls_output, labels)
        loss = alpha * loss_seg + beta * loss_cls

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_seg += loss_seg.item() * bs
        total_cls += loss_cls.item() * bs
        total_correct += (torch.argmax(cls_output, dim=1) == labels).sum().item()
        total_samples += bs

    return total_loss/total_samples, total_correct/total_samples, total_seg/total_samples, total_cls/total_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_dataset = LungDataset(args.train_csv, img_size=args.img_size)
    val_dataset = LungDataset(args.val_csv, img_size=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    model = ResNetSegmentationClassification(num_classes=4, pretrained=args.pretrained).to(device)
    criterion_seg, criterion_cls = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        train_loss, train_acc, train_seg, train_cls = train_one_epoch(model, train_loader, optimizer, criterion_seg, criterion_cls, device)
        val_loss, val_acc, val_seg, val_cls = validate_one_epoch(model, val_loader, criterion_seg, criterion_cls, device)

        print(f"Train -> Loss:{train_loss:.4f} | Acc:{train_acc:.4f} | Seg:{train_seg:.4f} | Cls:{train_cls:.4f}")
        print(f" Val  -> Loss:{val_loss:.4f} | Acc:{val_acc:.4f} | Seg:{val_seg:.4f} | Cls:{val_cls:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"[INFO] ✅ Best model saved with acc={val_acc:.4f}")

        scheduler.step()

    torch.save(model.state_dict(), "checkpoints/model_epoch_final.pth")
    print("[INFO] Final model saved!")

if __name__ == "__main__":
    main()
