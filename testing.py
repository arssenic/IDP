import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd

# Load your class labels
class_names = ['Stage1', 'Stage2', 'Stage3', 'Stage4']  # Adjust if needed

# Device configuration
device = torch.device('cpu')

# Image Transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load the trained VGG16 model
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(class_names))
model.load_state_dict(torch.load('best_vgg16.pth', map_location=device))
model = model.to(device)
model.eval()

# Folder containing test images
image_folder = 'impedimetry_images'  # Update path

# Store predictions
results = []

for img_name in os.listdir(image_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = class_names[predicted.item()]

        results.append({'Image_Name': img_name, 'Predicted_Label': predicted_label})
        print(f"{img_name}: {predicted_label}")

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('ripeness_predictions.csv', index=False)
print("Predictions saved to ripeness_predictions.csv")
