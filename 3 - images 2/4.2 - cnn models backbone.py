import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms, datasets
import numpy as np

import torch.nn as nn
import matplotlib.pyplot as plt

# Load pre-trained weights
vgg_model = vgg16(weights=VGG16_Weights.DEFAULT)

# Extract the input dimension
input_dim = nn.Sequential(*list(vgg_model.classifier.children()))[0].in_features

# Create a backbone with convolutional layers
backbone = nn.Sequential(*list(vgg_model.features.children()))

# Print the backbone model
print(backbone)
# Create a variable with the number of classes
num_classes = 2

# Create a sequential block
classifier = nn.Sequential(
    nn.Linear(input_dim, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes),
)

# Define the number of coordinates
num_coordinates = 4

bb = nn.Sequential(
    nn.Linear(input_dim, 32),
    nn.ReLU(),
    nn.Linear(32, num_coordinates),
)

# --- Visual Example ---

# Download a sample image from CIFAR10
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
img, label = dataset[0]

# Show the input image
plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
plt.title(f"Input Image - Label: {label}")
plt.axis('off')
plt.show()

# Add batch dimension and pass through backbone
img_batch = img.unsqueeze(0)
with torch.no_grad():
    features = backbone(img_batch)

# Visualize the first 8 feature maps from the last conv layer
feature_maps = features.squeeze(0)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[i].cpu().numpy(), cmap='viridis')
    ax.axis('off')
    ax.set_title(f'Feature map {i+1}')
plt.tight_layout()
plt.show()