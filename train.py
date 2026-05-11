from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------------
# Training settings:
# -----------------------------------
print("Setting up training settings")

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
print("Defining transformation functions")

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
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )]
)

# -----------------------------------
# Loading datasets
# -----------------------------------
print("Loading datasets")

# Transform raw images into datasets that undergo the above transformations
# where each image has their own correct label
# The dataset is a container like a java collection where each item in the
# collection is a 3 x 128 x 128 image and the label corresponding to that
# image
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transforms)
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transforms)

# Transforms each dataset into batches of smaller shuffled dataset that can be passed 
# into the model
# The loader is a container like a java collection where one item in the collection
# is a pair, where the first item in the pair is a collection of 32 images and the
# second item in the pair is a container containing their 32 corresponding labels
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------------
# Convolutional neural network
# -----------------------------------
print("Defining the CNN")

class CatCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Features is a sequential neural network model comprised of four
        # convolutional layers
        # nn.Sequential() returns an object whose type is:
        # torch.nn.modules.container.Sequential
        self.features = nn.Sequential(
            # First convolution layer - Learn simple features
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 32 x 128 x 128
            nn.MaxPool2d(2),
            # Output: 32 x 64 x 64

            # Second convolution layer - Learn intermediate features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 64 x 64 x 64
            nn.MaxPool2d(2),
            # Output: 64 x 32 x 32

            # Third convolution layer - Learn more intermediate features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 128 x 32 x 32
            nn.MaxPool2d(2),
            # Output: 128 x 16 x 16

            # Fourth convolution layer - Learn task-specific features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 256 x 16 x 16
            nn.MaxPool2d(2)
            # Output: 256 x 8 x 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Map the flattened feature maps of size 256 x 8 x 8 to 256 hidden neurons
            nn.Linear(256 * 8 * 8, 256),
            # ReLU for non-linear activation
            nn.ReLU(),
            # Randomly dropout the output from the hidden neurons to reduce risk of
            #overfitting
            nn.Dropout(0.4),
            # Final output layer with 2 hidden neurons
            nn.Linear(256, num_classes)
            # Note: The final output layer doesn't need softmax activation because 
            # PyTorch's cross entropy loss calculates the log-softmax from the
            # raw logits of the final linear output layer
        )

    # In Python, instance methods normally write self explicitly
    # Thus, we pass in the self inside the method parameters when we define them
    # In PyTorch, children of the nn.Module class needs to define and override
    # a forward method
    # In PyTorch, the forward() method is automatically called when the children of
    # the nn.Module class gets passed in a data from a loader, with the data
    # being the argument into the forward() method call
    def forward(self, x):
        # In Python, some objects can be called like functions
        # Objects are callable if their class defines a special method called __call__
        # That is the case for nn.Module, of which nn.Sequential is a child class
        # features is a type of nn.Sequential, so it can also be called like a
        # function
        x = self.features(x)
        x = self.classifier(x)
        return x
    
# -----------------------------------
# Define the model
# -----------------------------------
print("Defining the model variable")

# In PyTorch, the model is not just "code". It contains many tensors,
# and some of those tensors are learnable parameters. 
# Thus, this creates the model and stores the tensors in device memory
model = CatCNN(num_classes=2).to(device)

# -----------------------------------
# Loss
# -----------------------------------
print("Defining the loss function")

loss_fn = nn.CrossEntropyLoss()

# -----------------------------------
# Optimiser
# -----------------------------------
print("Defining the optimiser")

# Backward propagation only calculates the gradient
# PyTorch uses an optimiser to actually perform the weights updates
# Rather than using a simple optimiser, PyTorch allows us to use Adam
# Adam stands for Adaptive Moment Estimation and is an adaptive
# optimiser
# It keeps track of statistics such as the average magnitude of change
# in the weights recently and the direction of the change
# It then uses this information to answer questions like:
# 1. Has this weight been consistently changing in this direction?
# 2. Are the gradients for this weight usually large or small?
# 3. Should I take a bigger or smaller step for this weight update?
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------------
# Evaluation function
# -----------------------------------
print("Defining the evaluation function")

# The evaluation function assesses how well the model performs
# against test data
# .item() converts a single-value PyTorch tensor into a normal
# Python number
# By default, PyTorch CrossEntropyLoss() calculates the average
# loss
def eval_fn(loader):
    model.eval() 

    total_loss = 0
    correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in loader:
            # The labels and images need to be moved to device as well
            # because PyTorch needs labels, images, and model to be on
            # the same device for computation
            labels = labels.to(device)
            images = images.to(device)

            # Do the model computation
            outputs = model(images)

            # Do the model loss computation using raw logits
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)

            # Compute predictions and count correct predictions
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

            # Increment total images evaluated
            total_images += images.size(0)

    average_loss = total_loss / total_images
    accuracy = correct / total_images

    return average_loss, accuracy

# -----------------------------------
# Training Loop
# -----------------------------------
print("Starting training")

best_val_accuracy = 0

for epoch in range(EPOCHS):
    model.train()

    total_train_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimiser.step()

        total_train_loss += loss.item() * images.size(0)

    train_loss = total_train_loss / len(train_dataset)
    val_loss, val_accuracy = eval_fn(val_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        print("New best model found")
        torch.save(model.state_dict(), "best_cat_cnn.pth")

# -----------------------------------
# Final test
# -----------------------------------
print("Conduct final model test")

model.load_state_dict(torch.load("best_cat_cnn.pth", map_location=device))
test_loss, test_accuracy = eval_fn(test_loader)

print(f"Best validation accuracy: {best_val_accuracy} | ")
print(f"Test loss: {test_loss} | ")
print(f"Test accuracy: {test_accuracy}")