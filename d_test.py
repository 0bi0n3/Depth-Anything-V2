import cv2
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

print("Creating model...")
model = DepthAnythingV2(**model_configs[encoder])
print("Loading weights...")
model.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True))
model.to(DEVICE)
model.eval()

# Process an image
print("Loading image...")
image_path = './assets/unreal_street.jpg'  # Update this to your image path
if not os.path.exists(image_path):
    print(f"Image not found at {image_path}! Please provide a valid image path.")
    exit(1)

# Load and process the image
img = cv2.imread(image_path)
if img is None:
    print(f"Failed to load image at {image_path}")
    exit(1)

# Calculate new height that's divisible by 14
new_height = (img.shape[0] // 14) * 14  # 1078 for your case

# Calculate new width to maintain aspect ratio
aspect_ratio = img.shape[1] / img.shape[0]
new_width = int(new_height * aspect_ratio)
new_width = (new_width // 14) * 14  # Make width also divisible by 14

# Resize image
img = cv2.resize(img, (new_width, new_height))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Running inference...")
with torch.no_grad():
    # Preprocess image for the model
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    # Normalize if needed
    img_tensor = img_tensor / 255.0
    
    # Run inference
    depth = model(img_tensor)
    depth = depth.squeeze().cpu().numpy()

# Create output directory
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# Save the depth map
print("Saving results...")
plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(depth, cmap='plasma')
plt.title('Depth Map')
plt.axis('off')

plt.savefig(f'{output_dir}/depth_result.png')
print(f"Results saved to {output_dir}/depth_result.png")

# Also save as separate files
cv2.imwrite(f'{output_dir}/depth_map.png', (depth * 255).astype(np.uint8))
cv2.imwrite(f'{output_dir}/original.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))