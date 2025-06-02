import torch
import torch.nn as nn


# Define the ManufacturingCNN model class
class ManufacturingCNN(nn.Module):
    def __init__(self):
        super(ManufacturingCNN, self).__init__()
        # Example layers, modify as needed
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# Create and save the model
model = ManufacturingCNN()
torch.save(model.state_dict(), "ModelCNN.pth")

# Create a new model and load the saved state
loaded_model = ManufacturingCNN()
loaded_model.load_state_dict(torch.load("ModelCNN.pth"))
print(loaded_model)

# RES NET

from torchvision.models import resnet18, ResNet18_Weights

# Initialize model with default weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Set model to the evaluation mode
model.eval()

# Initialize the transforms
# Initialize the transforms
preprocess = weights.transforms()

# Dummy image for demonstration (replace with your actual image)
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.ops import box_iou

# Create a dummy grayscale image (224x224) for testing
img = Image.open("Espresso.jpeg").convert("RGB")

# Apply preprocessing transforms
batch = preprocess(img).unsqueeze(0)
# Apply model with softmax layer
prediction = model(batch).squeeze(0).softmax(0)

# Apply argmax
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(category_name)
# Define a dummy bbox for demonstration (replace with your actual bbox)
bbox = [50, 50, 150, 150]  # Example: [x_min, y_min, x_max, y_max]

# Convert bbox into tensors
bbox_tensor = torch.tensor(bbox)

# Add a new batch dimension
bbox_tensor = bbox_tensor.unsqueeze(0)
bbox_tensor = bbox_tensor.unsqueeze(0)

# Resize image and transform tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# Apply transform to image
image = img  # Use the previously loaded image
image_tensor = transform(image)
print(image_tensor)

# Apply transform to image
image_tensor = transform(image)
print(image_tensor)

import matplotlib.pyplot as plt

# Display the image with the bounding box
fig, ax = plt.subplots(1)
ax.imshow(img)
# Draw the bounding box
rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                     linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.title(f"Predicted: {category_name} ({score:.2f})")
plt.show()

# Convert bbox into tensors
bbox_tensor = torch.tensor(bbox)

# Add a new batch dimension
bbox_tensor = bbox_tensor.unsqueeze(0)

# Resize image and transform tensor
transform = transforms.Compose([
  transforms.Resize(224),
  transforms.PILToTensor()
])

# Apply transform to image
image_tensor = transform(image)
print(image_tensor)

