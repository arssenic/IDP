import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import h5py
import numpy as np

# Set device
device = torch.device("cpu")
print(f"Using device: {device}")

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets from folders
train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
val_dataset = datasets.ImageFolder(root="dataset/val", transform=transform)
test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)

class_names = train_dataset.classes
num_classes = len(class_names)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load Pretrained ResNet-50 Model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Move model to CPU
model = model.to(device)

# Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
num_epochs = 10
best_accuracy = 0.0
best_model_weights = None

for epoch in range(num_epochs):
    model.train()
    start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0

    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)

    for images, labels in train_progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        train_progress.set_postfix(loss=loss.item())

    train_accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    correct = 0
    total = 0

    val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)

    with torch.no_grad():
        for images, labels in val_progress:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    elapsed_time = time.time() - start_time

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, Time: {elapsed_time:.2f}s")

    # Save best model weights
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model_weights = model.state_dict()
        print("Best model updated!")

# Save best model to .h5 format
model.load_state_dict(best_model_weights)
model.eval()

with h5py.File("best_resnet50.h5", "w") as f:
    for name, param in model.state_dict().items():
        f.create_dataset(name, data=param.cpu().numpy())

print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

# ---- TESTING PHASE ----
correct = 0
total = 0
test_progress = tqdm(test_loader, desc="Testing", leave=True)

with torch.no_grad():
    for images, labels in test_progress:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        test_progress.set_postfix(accuracy=100 * correct / total)

test_accuracy = 100 * correct / total
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
