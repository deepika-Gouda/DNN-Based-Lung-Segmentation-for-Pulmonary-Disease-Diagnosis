# model/predict.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
import cv2
import os

from project_model.resnet_seg import ResNetSegmentationClassification

# -------------------------------
# Argument Parser
# -------------------------------
parser = argparse.ArgumentParser(description="Lung Segmentation and Disease Classification")
parser.add_argument('--image_path', type=str, default='test_xray.png', help='Path to input X-ray image')
parser.add_argument('--output_path', type=str, default='outputs/', help='Folder to save visualizations')
parser.add_argument('--checkpoint', type=str, default='checkpoints/model_epoch_final.pth', help='Path to trained model checkpoint')
args = parser.parse_args()

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -------------------------------
# Load Model
# -------------------------------
model = ResNetSegmentationClassification(num_classes=4, pretrained=False)
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint)
model.to(device).eval()
print(f"[INFO] Model loaded from: {args.checkpoint}")

# -------------------------------
# Preprocess Image
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if not os.path.exists(args.image_path):
    raise FileNotFoundError(f"❌ Input image not found at {args.image_path}")

original_image = Image.open(args.image_path).convert("RGB")
input_tensor = transform(original_image).unsqueeze(0).to(device)
input_tensor.requires_grad = True
original_np = np.array(original_image.resize((224, 224)))

# -------------------------------
# Grad-CAM setup
# -------------------------------
gradients = None
activations = None

def forward_hook(module, inp, out):
    global activations
    activations = out

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# Register hooks on last conv layer BEFORE forward pass
last_conv = None
for name, module in model.encoder.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        last_conv = module
if last_conv is None:
    raise ValueError("No Conv2d layer found in encoder for Grad-CAM hooks")

last_conv.register_forward_hook(forward_hook)
last_conv.register_full_backward_hook(backward_hook)

# -------------------------------
# Forward Pass
# -------------------------------
output = model(input_tensor)
seg_output = output["seg_mask"]        # [1,1,H,W]
cls_output = output["logits"]          # [1, num_classes]

# -------------------------------
# Postprocess Segmentation
# -------------------------------
seg_mask = torch.sigmoid(seg_output).detach().squeeze().cpu().numpy()
seg_mask_bin = (seg_mask > 0.5).astype(np.uint8) * 255

# -------------------------------
# Classification
# -------------------------------
class_names = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
pred_class_idx = torch.argmax(cls_output, dim=1).item()
pred_class_name = class_names[pred_class_idx]
print(f"[RESULT] Predicted Disease Class: {pred_class_name}")

# -------------------------------
# Grad-CAM computation
# -------------------------------
model.zero_grad()
cls_output[0, pred_class_idx].backward(retain_graph=True)

if gradients is None:
    raise RuntimeError("Gradients are None! Check hook registration and backward pass.")

weights = torch.mean(gradients, dim=(1,2), keepdim=True)
gradcam_map = torch.sum(weights * activations, dim=1).squeeze()
gradcam_map = F.relu(gradcam_map).detach().cpu().numpy()   # detach to avoid errors
gradcam_map = cv2.resize(gradcam_map, (224, 224))
gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() + 1e-8)

# Overlay Grad-CAM on original
gradcam_color = cv2.applyColorMap(np.uint8(255 * gradcam_map), cv2.COLORMAP_JET)
overlay_gc = cv2.addWeighted(original_np, 0.6, gradcam_color, 0.4, 0)

# -------------------------------
# Save visualizations
# -------------------------------
os.makedirs(args.output_path, exist_ok=True)
cv2.imwrite(os.path.join(args.output_path, "original_xray.png"), cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(args.output_path, "segmentation_mask.png"), seg_mask_bin)
cv2.imwrite(os.path.join(args.output_path, "gradcam_heatmap.png"), gradcam_color)
cv2.imwrite(os.path.join(args.output_path, f"overlay_{pred_class_name}.png"), overlay_gc)

print(f"[INFO] All visualizations saved in folder: {args.output_path}")
