# losses.py
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Helps handle class imbalance by focusing on hard-to-classify samples.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: [B, C] raw logits
        targets: [B] class indices
        """
        ce_loss = self.ce(inputs, targets)  # standard cross-entropy loss
        pt = torch.exp(-ce_loss)           # pt is the probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
