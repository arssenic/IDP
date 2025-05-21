import os
import shutil

# Set your source folder (with subfolders) and destination (optional)
source_folder = 'impedimetry_data'  # Replace with your path
destination_folder = 'impedimetry_images'  # Optional: folder to copy all jpgs

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Loop through all subdirectories and files
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.lower().endswith('.jpg'):
            full_path = os.path.join(root, file)
            print(f"Found: {full_path}")
            
            # Optional: Copy to destination folder
            shutil.copy(full_path, destination_folder)
