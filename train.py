from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------------
# Training settings:
# -----------------------------------

# Constants
DATA_DIR = Path("data")
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

#Processor choice 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# Transformation functions
# -----------------------------------

# Image data transformation for training
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Image data transformation for evaluation
eval_transforms = transforms.Compose(
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
)

# -----------------------------------
# Loading datasets
# -----------------------------------

# Transform raw images into datasets that undergo the above transformations
# where each image has their own correct label
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transforms)
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transforms)

# Transforms each dataset into batches of smaller shuffled dataset that can be passed 
# into the model
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------------
# Convolutional neural network
# -----------------------------------

class CatCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Features is a sequential neural network model comprised of four
        # convolutional layers
        self.features = nn.Sequential(
            # First convolution layer - Learn simple features
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.RelU(),
            # Output: 32 x 128 x 128
            nn.MaxPool2d(2),
            # Output: 32 x 64 x 64

            # Second convolution layer - Learn intermediate features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.RelU(),
            # Output: 64 x 64 x 64
            nn.MaxPool2d(2),
            # Output: 64 x 32 x 32

            # Third convolution layer - Learn more intermediate features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.RelU,
            # Output: 128 x 32 x 32
            nn.MaxPool2d(2),
            # Output: 128 x 16 x 16

            # Fourth convolution layer - Learn task-specific features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU,
            # Output: 256 x 16 x 16
            nn.MaxPool2d(2)
            # Output: 256 x 8 x 8
        )


        


