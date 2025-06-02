F.get_image_num_channels()

# Create a model
model = CNNModel()
print("Original model: ", model)

# Create a new convolutional layer
conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

# Append the new layer to the model
model.add_module("conv2", conv2)
print("Extended model: ", model)


class BinaryImageClassification(nn.Module):
    def __init__(self):
        super(BinaryImageClassification, self).__init__()
        # Create a convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # Pass inputs through the convolutional block
        x = self.conv_block(x)
        return x
