from pathlib import Path
import random
import shutil 

# Declare the input and output paths
IN_ROOT = Path("archive/raw-img")
OUT_ROOT = Path("data")

# Create dict to categorise into cat and not_cat
folder_to_label = {
    "gatto": "cat",
    "cane": "not_cat",
    "cavallo": "not_cat",
    "elefante": "not_cat",
    "farfalla": "not_cat",
    "gallina": "not_cat",
    "mucca": "not_cat",
    "pecora": "not_cat",
    "ragno": "not_cat",
    "scoiattolo": "not_cat"
}

# Create the folder structure
for split in ["train", "val", "test"]:
    for label in ["cat", "not_cat"]:
        folder = OUT_ROOT / split / label
        folder.mkdir(parents=True, exist_ok=True)

# Assign the images into their respective image lists
cat_images = []
not_cat_images = []
for animal_folder in IN_ROOT.iterdir():
    label = folder_to_label[animal_folder.name]
    for image_path in animal_folder.iterdir():
        if (label == "cat"):
            cat_images.append(image_path)
        else:
            not_cat_images.append(image_path)
    
# Shuffle the image order
random.shuffle(cat_images)
random.shuffle(not_cat_images)

# Cut down the length of the not_cat images to the same as cat images
n = len(cat_images)
not_cat_images = not_cat_images[:n]

# Split into the respective train, val, and test lists
train_end = int(0.7 * n)
val_end = int(0.85 * n)

cat_train = cat_images[:train_end]
cat_val = cat_images[train_end:val_end]
cat_test = cat_images[val_end:]

not_cat_train = not_cat_images[:train_end]
not_cat_val = not_cat_images[train_end:val_end]
not_cat_test = not_cat_images[val_end:]

# Copy the images from the original folder to the new folder structure
for image_path in cat_train:
    destination_name = image_path.parent.name + "_" + image_path.name
    destination_path = OUT_ROOT / "train" / "cat" / destination_name
    shutil.copy2(image_path, destination_path)

for image_path in cat_val:
    destination_name = image_path.parent.name + "_" + image_path.name
    destination_path = OUT_ROOT / "val" / "cat" / destination_name
    shutil.copy2(image_path, destination_path)

for image_path in cat_test:
    destination_name = image_path.parent.name + "_" + image_path.name
    destination_path = OUT_ROOT / "test" / "cat" / destination_name
    shutil.copy2(image_path, destination_path)

for image_path in not_cat_train:
    destination_name = image_path.parent.name + "_" + image_path.name
    destination_path = OUT_ROOT / "train" / "not_cat" / destination_name
    shutil.copy2(image_path, destination_path)

for image_path in not_cat_val:
    destination_name = image_path.parent.name + "_" + image_path.name
    destination_path = OUT_ROOT / "val" / "not_cat" / destination_name
    shutil.copy2(image_path, destination_path)

for image_path in not_cat_test:
    destination_name = image_path.parent.name + "_" + image_path.name
    destination_path = OUT_ROOT / "test" / "not_cat" / destination_name
    shutil.copy2(image_path, destination_path)

