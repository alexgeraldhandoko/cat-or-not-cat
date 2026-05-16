# This file defines important functions to use the model

from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

from .cnn import CatCNN

IMG_SIZE = 128

CAT_LABELS = ["cat", "not_cat"]

# Define the transformation function used by the model in
# real prediction task
def build_image_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

# Returns the model and device to be used during prediction task
def load_model(model_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatCNN(num_classes=2)
    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict=state_dict)
    model.to(device)
    model.eval()

    return model, device

# Use the model to predict the class of the image
def predict_image(image: Image.Image, model: CatCNN, device: torch.device):
    transform = build_image_transform()
    image_tensor = transform(image)
    # Add a 1 to the front of the image_tensor shape since the model later
    # expects a 4 dim shape tensor since it expects a batch of images
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    # image tensor shape: 1 x 3 x 128 x 128

    with torch.no_grad():
        outputs = model(image)
        # output shape: 1 x 2, since it processes results for all the 
        # batches, which is just one batch in this case
        probabilities = torch.softmax(outputs, dim=1)
        confidence, index = torch.max(probabilities, dim=1)
    
    label = CAT_LABELS[index]

    return {
        "label": label,
        "confidence": confidence
    }



