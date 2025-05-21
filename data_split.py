import os
import shutil
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Source dataset path (with Stage1, Stage2, etc. subfolders)
source_dir = 'total_images'
output_base = 'splitted_data'

# Create target folders
for split in ['train', 'val', 'test']:
    for stage in ['Stage1', 'Stage2', 'Stage3', 'Stage4']:
        os.makedirs(os.path.join(output_base, split, stage), exist_ok=True)

# Gather all images and their labels
image_paths = []
labels = []

for stage in os.listdir(source_dir):
    stage_path = os.path.join(source_dir, stage)
    if os.path.isdir(stage_path):
        for img_file in os.listdir(stage_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(stage_path, img_file))
                labels.append(stage)

# Split with stratification (preserve label distribution)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels, random_state=42
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# Function to copy images to split folders
def copy_images(paths, labels, split_name):
    for path, label in zip(paths, labels):
        dest_dir = os.path.join(output_base, split_name, label)
        shutil.copy(path, dest_dir)

copy_images(train_paths, train_labels, 'train')
copy_images(val_paths, val_labels, 'val')
copy_images(test_paths, test_labels, 'test')

print("Stratified and balanced dataset split completed.")
