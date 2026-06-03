# model/resnet_seg.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetSegmentationClassification(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNetSegmentationClassification, self).__init__()

        # Load backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # exclude avgpool & fc

        # Segmentation Head (U-Net style upsampling)
        self.seg_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(32, 1, 1)  # final mask
        )

        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.encoder(x)

        # Segmentation
        seg_logits = self.seg_head(features)
        seg_logits = F.interpolate(seg_logits, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)

        # Classification
        pooled = self.avgpool(features).flatten(1)
        logits = self.classifier(pooled)

        return {"seg_mask": seg_logits, "logits": logits}
