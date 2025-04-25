import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Path to the pre-trained MaxxVit model
model_path = r'C:\GitFolders\fabric_models/models/coatnet/coatnet.pt'

# Path to the cotton fabric image
image_path = r'C:\GitFolders\fabric_models\data\fabrics_clean\Cotton\Cotton_50_1.png'

# Choose the target layer for attention map visualization (we will find it by type)
target_layer_name = 'attn'
target_block_type = timm.models.maxxvit.TransformerBlock2d

# Load the MaxxVit model
model_name = "coatnet_0_rw_224.sw_in1k"
num_classes = 20  # Assuming your model was trained for 20 classes
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the image
try:
    img = Image.open(image_path).convert('RGB')
except FileNotFoundError:
    print(f"Error: Image not found at {image_path}")
    exit()

# Apply transformations and add batch dimension
input_tensor = transform(img).unsqueeze(0)

# Hook to get the output of the target layer
attention_map = None
def hook_fn(module, input, output):
    global attention_map
    attention_map = output.detach().cpu().numpy()

# Find the target layer by iterating through the blocks
found_layer = False
for stage in model.stages:
    for block in stage.blocks:
        if isinstance(block, target_block_type):
            if hasattr(block, target_layer_name):
                target_module = getattr(block, target_layer_name)
                hook = target_module.register_forward_hook(hook_fn)
                found_layer = True
                print(f"Hook registered for layer: {type(block).__name__}.{target_layer_name}")
                break  # Break after finding the first TransformerBlock2d
    if found_layer:
        break

if not found_layer:
    print(f"Error: Could not find {target_block_type.__name__} with attribute '{target_layer_name}'.")
    exit()

# Perform inference
with torch.no_grad():
    model(input_tensor)

# Remove the hook
hook.remove()

if attention_map is not None:
    # Extract the attention map (e.g., sum across the head dimension)
    attention_map = np.sum(attention_map[0], axis=1)  # Sum across the head dimension

    # Normalize the attention map
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))

    # Resize the attention map to the original image size
    resized_attention_map = np.array(Image.fromarray(attention_map).resize(img.size))

    # Create a heatmap
    heatmap = plt.cm.jet(resized_attention_map)[:, :, :3]

    # Convert the original image to numpy array and normalize
    img_np = np.array(img) / 255.0

    # Overlay the heatmap on the original image
    overlayed_image = 0.6 * img_np + 0.4 * heatmap

    # Display the result with three subplots
    plt.figure(figsize=(15, 5))

    # Subplot 1: Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # Subplot 2: Attention Map (Heatmap)
    plt.subplot(1, 3, 2)
    plt.imshow(resized_attention_map, cmap='jet')
    plt.title(f'Attention Map (Layer: {target_block_type.__name__}.{target_layer_name})')
    plt.axis('off')

    # Subplot 3: Attention Map Overlaid
    plt.subplot(1, 3, 3)
    plt.imshow(overlayed_image)
    plt.title('Attention Map Overlaid')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('maxxvit_attention_map_subplots.png') # Or plt.show() to display
else:
    print(f"Error: Feature map for layer {target_layer_name} not found.")