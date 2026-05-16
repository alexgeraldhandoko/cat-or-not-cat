from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ..cnn import CatCNN

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

# Resume training option
BEST_MODEL_PATH = Path("best_cat_cnn.pth")
CHECKPOINT_PATH = Path("last_checkpoint.pth")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training from last_checkpoint.pth if it exists."
)
args = parser.parse_args()

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
# Resume Training If Requested
# -----------------------------------

start_epoch = 0
best_val_accuracy = -1.0

if args.resume and CHECKPOINT_PATH.exists():
    print(f"Resuming training from {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_accuracy = checkpoint["best_val_accuracy"]

    print(f"Already trained for {start_epoch} epochs.")
    print(f"Previous best validation accuracy: {best_val_accuracy:.4f}")
elif args.resume:
    print("Resume requested, but no checkpoint was found. Starting model training from scratch.")
else:
    print("Starting model training from scratch")


# -----------------------------------
# Training Loop
# -----------------------------------

print("Starting training")

end_epoch = start_epoch + EPOCHS

for epoch in range(start_epoch, end_epoch):
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
        f"Epoch {epoch + 1}/{end_epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    # Store the best model so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        print("New best model found")
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    
    # Save the most recent model into the checkpoint file
    # every epoch, even if it doesn't yield the best val
    # accuracy since we want to save progress of the most
    # recent epoch into the checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "class_to_idx": train_dataset.class_to_idx
        },
        CHECKPOINT_PATH
    )

# -----------------------------------
# Final test
# -----------------------------------
print("Conduct final model test")

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
test_loss, test_accuracy = eval_fn(test_loader)

print(f"Best validation accuracy: {best_val_accuracy} | ")
print(f"Test loss: {test_loss} | ")
print(f"Test accuracy: {test_accuracy}")