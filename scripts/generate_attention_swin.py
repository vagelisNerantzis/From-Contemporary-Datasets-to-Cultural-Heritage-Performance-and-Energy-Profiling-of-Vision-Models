import argparse
import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

attention_maps = []
handles = []

def attn_hook(module, input, output):
    print(f"Input shape to {module.__class__.__name__}.forward: {input[0].shape}")
    # Let's try to access the attention weights here
    try:
        # Get the input tensor
        x = input[0]
        B, H, W, C = x.shape

        # Apply the qkv linear layer
        qkv = module.qkv(x)  # Shape: (B, H, W, 3 * num_heads * head_dim)
        num_heads = module.num_heads
        head_dim = C // num_heads

        # Reshape qkv to separate Q, K, V
        qkv = qkv.reshape(B, H * W, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: (B, H*W, num_heads, head_dim)

        # Transpose for attention calculation
        q = q.transpose(1, 2)  # Shape: (B, num_heads, H*W, head_dim)
        k = k.transpose(1, 2)  # Shape: (B, num_heads, H*W, head_dim)
        v = v.transpose(1, 2)  # Shape: (B, num_heads, H*W, head_dim)

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)  # Shape: (B, num_heads, H*W, H*W)

        # Apply softmax
        attn = attn.softmax(dim=-1)

        attention_maps.append(attn.detach().cpu())
        print(f"Captured attention map shape: {attn.shape}")

    except Exception as e:
        print(f"Error in attn_hook: {e}")

def generate_swin_attention(model, image_path, output_path, device):
    global attention_maps
    attention_maps = []
    global handles
    handles = []

    def hook_fn(module, input, output):
        if isinstance(module, models.swin_transformer.SwinTransformerBlock):
            for name, submodule in module.named_modules():
                if name == "attn" and isinstance(submodule, models.swin_transformer.ShiftedWindowAttention):
                    handle = submodule.register_forward_hook(attn_hook)
                    handles.append(handle)

    # Register hooks
    for name, module in model.named_modules():
        hook_fn(module, None, None)

    # Load the image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Remove hooks
    for handle in handles:
        handle.remove()

    print(f"Number of attention maps captured: {len(attention_maps)}")
    if not attention_maps:
        raise RuntimeError("No attention maps captured from Swin Transformer.")

    # Process and save attention maps
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Visualize the attention map from the last Swin Transformer Block
    last_attn_map = attention_maps[-1]  # Shape: [1, num_heads, H*W, H*W]
    num_heads = last_attn_map.shape[1]
    spatial_size = int(np.sqrt(last_attn_map.shape[2]))  # e.g., 7 for the last block

    # Average across all attention heads
    avg_attn_map = torch.mean(last_attn_map, dim=1).squeeze(0) # Shape: [H*W, H*W]

    # Attention from the first token to all other tokens
    attn_weights = avg_attn_map[0, :].reshape(spatial_size, spatial_size) # Shape: [spatial_size, spatial_size]

    # Upsample the attention map to the original image size
    upsample_layer = transforms.Resize(original_size[::-1], interpolation=transforms.InterpolationMode.BILINEAR) # Reverse size for PIL (width, height)
    upsampled_attn_map = upsample_layer(attn_weights.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).cpu().numpy()

    # Normalize the attention map
    upsampled_attn_map = (upsampled_attn_map - np.min(upsampled_attn_map)) / (np.max(upsampled_attn_map) - np.min(upsampled_attn_map))

    # Visualize the attention map
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # Attention map
    plt.subplot(1, 3, 2)
    plt.imshow(upsampled_attn_map, cmap='jet')
    plt.title("Attention Map (Last Block, Averaged Heads, From First Token)")
    plt.axis('off')

    # Overlayed attention map
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(upsampled_attn_map, cmap='jet', alpha=0.5)
    plt.title("Attention Map Overlayed")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Attention map saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate attention maps for Swin Transformer.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained Swin Transformer model.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output_path', type=str, default='output/attention_maps.png', help='Path to save the output attention maps.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
    args = parser.parse_args()

    # Load the Swin Transformer model from torchvision
    model = models.swin_t(weights=None) # Initialize the model without pretrained weights

    # Modify the classification head to match the number of classes in your trained model (20)
    num_classes = 20
    model.head = torch.nn.Linear(model.head.in_features, num_classes)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval().to(args.device)

    generate_swin_attention(model, args.image_path, args.output_path, args.device)

if __name__ == '__main__':
    main()