# import os
# import shutil
# from sklearn.model_selection import train_test_split

# # Define the paths
# base_dir = 'd:/PixelPen-Model/dataset'
# train_dir = os.path.join(base_dir, 'train')
# val_dir = os.path.join(base_dir, 'validation')

# # Create train and validation directories if they don't exist
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)

# # Get the list of class directories (000, 001, etc.)
# class_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# # Split data and move files
# for class_dir in class_dirs:
#     class_path = os.path.join(base_dir, class_dir)
#     images = os.listdir(class_path)
#     train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    
#     # Create class subdirectories in train and validation directories
#     train_class_dir = os.path.join(train_dir, class_dir)
#     val_class_dir = os.path.join(val_dir, class_dir)
#     os.makedirs(train_class_dir, exist_ok=True)
#     os.makedirs(val_class_dir, exist_ok=True)
    
#     # Move images to the respective directories
#     for img in train_images:
#         shutil.move(os.path.join(class_path, img), os.path.join(train_class_dir, img))
#     for img in val_images:
#         shutil.move(os.path.join(class_path, img), os.path.join(val_class_dir, img))
#====================================================================================================


import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

# Define the paths
base_dir = 'd:/PixelPen-Model/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# Create train and validation directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Function to check if a file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

# Iterate through all directories and subdirectories to find images
image_paths = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if is_image_file(file):
            image_paths.append(os.path.join(root, file))

# Split data and move files
for img_path in image_paths:
    # Get the class directory (e.g., '000', '001', etc.)
    class_dir = os.path.basename(os.path.dirname(img_path))
    
    # Split data into train and validation sets
    if 'train' in img_path:
        target_dir = train_dir
    else:
        target_dir = val_dir
    
    # Create class subdirectories in train and validation directories
    class_target_dir = os.path.join(target_dir, class_dir)
    os.makedirs(class_target_dir, exist_ok=True)
    
    # Move images to the respective directories
    shutil.move(img_path, os.path.join(class_target_dir, os.path.basename(img_path)))
