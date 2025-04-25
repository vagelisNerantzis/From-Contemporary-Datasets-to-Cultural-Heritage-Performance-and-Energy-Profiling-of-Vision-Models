import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms
from PIL import Image
import cv2

# ======================= PATHS =======================
image_path = r"C:\GitFolders\fabric_models\data\fabrics_clean\Cotton\Cotton_50_1.png"
model_path = r"C:\GitFolders\fabric_models\models\resnet50\resnet50.pt"
output_path = r"C:\GitFolders\fabric_models\scripts\resnet50_heatmap_grid.png" 

# ======================= DEVICE =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= LOAD MODEL =======================
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 20)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ======================= LAST CONV LAYER =======================
target_layer = model.layer4[-1].conv3
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

handle_fwd = target_layer.register_forward_hook(forward_hook)
handle_bwd = target_layer.register_backward_hook(backward_hook)

# ======================= PREPROCESS IMAGE =======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# ======================= FORWARD + BACKWARD =======================
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()
class_score = output[0, pred_class]
model.zero_grad()
class_score.backward()

# ======================= GRAD-CAM COMPUTATION =======================
grads_val = gradients[0].cpu().data.numpy()[0]
fmap = features[0].cpu().data.numpy()[0]
weights = np.mean(grads_val, axis=(1, 2))

cam = np.zeros(fmap.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * fmap[i, :, :]
cam = np.maximum(cam, 0)
cam = cam / cam.max()
cam = cv2.resize(cam, (224, 224))

# ======================= CONVERT TO VISUALS =======================
img_np = np.array(image.resize((224, 224)))
heatmap = (cam * 255).astype(np.uint8)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)

# ======================= MATPLOTLIB GRID OUTPUT =======================
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(img_np)
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(heatmap_color)
axs[1].set_title("Grad-CAM Heatmap")
axs[1].axis("off")

axs[2].imshow(overlay)
axs[2].set_title("Overlay (Image + Heatmap)")
axs[2].axis("off")

plt.tight_layout()
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.close()

print(f"✅ Saved heatmap visualization to {output_path}")

# Remove hooks
handle_fwd.remove()
handle_bwd.remove()
