import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms
from PIL import Image
import os
import argparse
import sys
import matplotlib.cm as cm  # Import the colormap module

class AttentionCapturer:
    def __init__(self):
        self.attentions = []

    def __call__(self, module, module_input, module_output):
        # Extract attention matrix from the attention block
        if isinstance(module_output, tuple):
            output = module_output[0]
        else:
            output = module_output

        # Try to access attention weights
        if hasattr(module, 'attn'):
            if hasattr(module.attn, 'attention_weights'):
                self.attentions.append(module.attn.attention_weights.detach())
            elif hasattr(module, 'attn_drop') and hasattr(module.attn_drop, 'attention_weights'):
                self.attentions.append(module.attn_drop.attention_weights.detach())

        # If none of the above worked, we'll try to monkey patch
        if len(self.attentions) == 0 and hasattr(module, 'attn'):
            original_forward = module.attn.forward

            def attention_forward_hook(self, *args, **kwargs):
                output = original_forward(*args, **kwargs)
                if isinstance(output, tuple) and len(output) > 1:
                    self.attentions.append(output[1].detach())
                return output

            module.attn.forward = attention_forward_hook.__get__(module.attn, type(module.attn))

def generate_attention_map(model, image_path, output_path, device='cpu'):
    """Generate attention map for a given image using the trained ViT model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load image
    print(f"Loading image from: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Modified approach to capture attention
    attention_capturer = AttentionCapturer()

    # Register hooks for all transformer blocks in the encoder
    hooks = []
    for i, block in enumerate(model.encoder.layers):
        # Try to access self_attention or self-attention based on PyTorch version
        attention_module = getattr(block, 'self_attention', None)
        if attention_module is None:
            attention_module = getattr(block, 'self_attention', None)
        if attention_module is None:
            attention_module = getattr(block, 'attn', None)

        if attention_module is not None:
            hooks.append(attention_module.register_forward_hook(attention_capturer))
        else:
            print(f"Warning: Could not find attention module in block {i}")

    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # If we still don't have attention weights, we need to try a different approach
    if len(attention_capturer.attentions) == 0:
        print("Standard attention capture failed. Trying alternative approach using hooks on encoder blocks...")

        all_attentions = []

        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor):
                all_attentions.append(output[1].detach())
            elif hasattr(output, 'attentions'):
                all_attentions.append(output.attentions.detach())
            elif hasattr(module, 'attn') and hasattr(module.attn, 'attention_weights'):
                all_attentions.append(module.attn.attention_weights.detach())
            elif hasattr(module, 'attn_drop') and hasattr(module.attn_drop, 'attention_weights'):
                all_attentions.append(module.attn_drop.attention_weights.detach())

        hooks = []
        for i, block in enumerate(model.encoder.layers):
            if hasattr(block, 'attn'):
                hooks.append(block.attn.register_forward_hook(attention_hook))
            elif hasattr(block, 'self_attention'): # Try this name as well
                hooks.append(block.self_attention.register_forward_hook(attention_hook))
            elif hasattr(block, 'self-attention'): # And this one
                hooks.append(block['self-attention'].register_forward_hook(attention_hook))


        # Run forward pass again
        with torch.no_grad():
            _ = model(input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if not all_attentions:
            # If still no attentions, we'll try a completely different approach
            # Let's synthesize attention using the gradients with respect to the image
            print("Hook-based approach failed. Trying gradient-based attention...")
            model.zero_grad()
            input_tensor.requires_grad_(True)

            # Get the output class prediction
            with torch.set_grad_enabled(True):
                output = model(input_tensor)
                predicted_class = output.argmax(1)

                # Calculate gradients for the predicted class
                output[0, predicted_class].backward()

            # Use gradients as a proxy for attention
            gradients = input_tensor.grad.abs()
            attention_proxy = gradients.sum(dim=1).squeeze(0)  # Sum across channels

            # Normalize and reshape to match original image dimensions
            attention_proxy = attention_proxy.cpu().numpy()
            attention_proxy = (attention_proxy - attention_proxy.min()) / (attention_proxy.max() - attention_proxy.min() + 1e-8)

            # We don't have attention on the CLS token, so we'll just use this directly
            cls_attn = attention_proxy
        else:
            # Use the last layer's attention weights
            last_attn = all_attentions[-1]  # [batch, heads, tokens, tokens]

            # Average over attention heads
            attn = last_attn[0].mean(0)  # [tokens, tokens]

            # Get attention from CLS token to all patch tokens
            cls_attn = attn[0, 1:]  # [num_patches]

            # Reshape to grid (14x14 for 224x224 images with ViT-B/16)
            grid_size = int(np.sqrt(cls_attn.size(0)))
            cls_attn = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
    else:
        # Process the captured attention weights
        last_attn = attention_capturer.attentions[-1]  # [batch, heads, tokens, tokens]

        # Average over attention heads
        attn = last_attn[0].mean(0)  # [tokens, tokens]

        # Get attention from CLS token to all patch tokens
        cls_attn = attn[0, 1:]  # [num_patches]

        # Reshape to grid (14x14 for 224x224 images with ViT-B/16)
        grid_size = int(np.sqrt(cls_attn.size(0)))
        cls_attn = cls_attn.reshape(grid_size, grid_size).cpu().numpy()

    # Normalize attention weights to [0, 1]
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    # Create attention map image with colormap
    cmap = cm.get_cmap('coolwarm')  # Changed colormap to 'coolwarm'
    attn_colormap = cmap(cls_attn)
    # Remove the alpha channel if it exists (shape (H, W, 4))
    if attn_colormap.shape[-1] == 4:
        attn_colormap = attn_colormap[:, :, :3]

    # Create overlay
    overlay_image = np.array(image.resize((224, 224))).astype(float) / 255

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(overlay_image)  # Use the normalized original image
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(attn_colormap) # Use the colormapped attention map
    axes[1].set_title("Attention Map")
    axes[1].axis('off')

    # Overlay the attention map on the original image with transparency
    axes[2].imshow(overlay_image)
    axes[2].imshow(attn_colormap, alpha=0.5, extent=(0, 224, 224, 0)) # Adjust extent if needed
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Attention map saved to {output_path}")

    return cls_attn, overlay_image

def main():
    # Add project root to path for correct relative paths
    # Get the absolute path of the script
    script_path = os.path.abspath(__file__)
    # Go up one directory level to reach the project root
    project_root = os.path.dirname(os.path.dirname(script_path))
    # Add to sys.path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(description="Generate attention maps from a ViT model")
    parser.add_argument("--model-path", type=str, default=os.path.join(project_root, "models/vit/vit.pt"),
                        help="Path to the saved model")
    parser.add_argument("--image-path", type=str,
                        default=os.path.join(project_root, "data/fabrics_clean/Cotton/Cotton_50_1.png"),
                        help="Path to the input image")
    parser.add_argument("--output-path", type=str,
                        default=os.path.join(project_root, "scripts/vit_attentionmap_grid.png"),
                        help="Path to save the attention map visualization")
    parser.add_argument("--num-classes", type=int, default=20,
                        help="Number of classes the model was trained on")
    parser.add_argument("--weights-only", action="store_true",
                        help="Use weights_only=True when loading the model")

    args = parser.parse_args()

    print(f"Project root: {project_root}")
    print(f"Model path: {args.model_path}")
    print(f"Image path: {args.image_path}")
    print(f"Output path: {args.output_path}")

    # Check if paths exist
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Attempting to save attention map to: {args.output_path}")  # Added print statement

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = models.vit_b_16(weights=None)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, args.num_classes)

    # Address the FutureWarning by setting weights_only=True
    if args.weights_only:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    model.to(device)
    model.eval()

    # Generate attention map
    generate_attention_map(model, args.image_path, args.output_path, device)

if __name__ == "__main__":
    main()