from torchvision.datasets import ImageFolder
from torchvision import transforms

train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ]
)

dataset_train = ImageFolder(
    "images/train/",
    transform=train_transforms,
)
