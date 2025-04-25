import argparse
import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("Ξεκινάει το script generate_convnext_attention.py")

feature_maps = []
handles = []

def hook_fn(module, input, output):
    print(f"Output shape of feature map from {module.__class__.__name__}: {output.shape}")
    feature_maps.append(output.detach().cpu())

def generate_convnext_attention(model, image_path, output_path, device):
    global feature_maps
    feature_maps = []
    global handles
    handles = []

    print("Πριν τη φόρτωση του μοντέλου")
    # Θα στοχεύσουμε σε ένα από τα τελευταία convolutional layers.
    # Η ακριβής ονομασία του layer μπορεί να χρειαστεί προσαρμογή ανάλογα με την αρχιτεκτονική.
    print(model.features[-1])  # <--- Πρόσθεσε αυτή τη γραμμή
    target_layer = model.features[-1][0].block[0]
    
    handle = target_layer.register_forward_hook(hook_fn)
    handles.append(handle)

    print("Μοντέλο φορτώθηκε")

    print("Πριν τη φόρτωση της εικόνας")
    # Load the image and prepare it for the model
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Συνήθως τα ConvNeXt εκπαιδεύονται σε 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    print("Εικόνα φορτώθηκε")

    print("Πριν το forward pass")
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    print("Μετά το forward pass")

    # Remove hooks
    for handle in handles:
        handle.remove()

    print(f"Number of feature maps captured: {len(feature_maps)}")
    if not feature_maps:
        raise RuntimeError("No feature maps captured from ConvNeXt.")

    # Process and save attention maps
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Πάρουμε το feature map από το τελευταίο layer
    last_feature_map = feature_maps[0] # Συνήθως έχουμε μόνο ένα captured feature map εδώ

    # Μέσο-ποίηση κατά μήκος των καναλιών
    attention_map = torch.mean(last_feature_map, dim=1, keepdim=True)

    # Αναβάθμιση στο μέγεθος της αρχικής εικόνας
    upsample_layer = transforms.Resize(original_size[::-1], interpolation=transforms.InterpolationMode.BILINEAR)
    upsampled_attention_map = upsample_layer(attention_map).squeeze().cpu().numpy()

    # Normalize
    upsampled_attention_map = (upsampled_attention_map - np.min(upsampled_attention_map)) / (np.max(upsampled_attention_map) - np.min(upsampled_attention_map))

    # Visualize
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # Attention map
    plt.subplot(1, 3, 2)
    plt.imshow(upsampled_attention_map, cmap='jet')
    plt.title("ConvNeXt Attention Map")
    plt.axis('off')

    # Overlayed attention map
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(upsampled_attention_map, cmap='jet', alpha=0.5)
    plt.title("Attention Map Overlayed")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Attention map saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate attention maps for ConvNeXt.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained ConvNeXt model.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output_path', type=str, default='output/convnext_attention_maps.png', help='Path to save the output attention maps.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
    args = parser.parse_args()

    # Load the ConvNeXt model
    model = models.convnext_tiny(weights=None) # Αρχικοποίηση χωρίς προ-εκπαιδευμένα βάρη

    print("Πριν τη φόρτωση των weights του μοντέλου")
    # Αντικατάσταση του classification head με βάση τον αριθμό των κλάσεων (20)
    num_classes = 20
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    print("Weights του μοντέλου φορτώθηκαν")
    model.eval().to(args.device)

    generate_convnext_attention(model, args.image_path, args.output_path, args.device)

    print("Το script main ολοκληρώθηκε")

if __name__ == '__main__':
    main()