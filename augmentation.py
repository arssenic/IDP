import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from tqdm import tqdm
import shutil

# Define augmentation pipeline with more variations
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flip with 50% probability
    iaa.Flipud(0.2),  # Vertical flip with 20% probability
    iaa.Affine(rotate=(-30, 30), scale=(0.8, 1.2)),  # Random rotation and scaling
    iaa.Multiply((0.7, 1.3)),  # Adjust brightness
    iaa.GaussianBlur(sigma=(0, 1.5)),  # Apply blur
    iaa.AdditiveGaussianNoise(scale=(0, 10)),  # Add noise
    iaa.LinearContrast((0.75, 1.5)),  # Adjust contrast
    iaa.Crop(percent=(0, 0.1))  # Random cropping
])

# Paths
input_dir = "DATA"  # Original dataset directory
output_dir = "AUGMENTED_DATA"  # Augmented dataset directory

# Remove old augmented data and create a fresh folder
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Process each class folder
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    images = [img for img in os.listdir(class_path) if img.endswith((".jpg", ".png"))]

    for img_name in tqdm(images, desc=f"Augmenting {class_name}"):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Save original image
        cv2.imwrite(os.path.join(output_class_path, img_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Generate **10 augmented versions** per original image
        for i in range(10):
            aug_img = augmenters(image=img)
            aug_name = f"{img_name.split('.')[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(output_class_path, aug_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

print("Data Augmentation Complete! Augmented images saved in", output_dir)
