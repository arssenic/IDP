import os
import numpy as np
import cv2
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

# Define dataset path
data_dir = "DATA"
categories = ["Decay", "Green", "Overripe", "Ripe"]  # Ensure order is correct

# Function to extract color histogram features
def extract_features(image_path, bins=(16, 16, 16)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # Resize for consistency
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)  # Normalize histogram
    return hist.flatten()

# Load data
features = []
labels = []

for category in categories:
    path = os.path.join(data_dir, category)
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist. Skipping...")
        continue
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        features.append(extract_features(img_path))
        labels.append(category)

features = np.array(features)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Check class order
print("Class mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

# Check class distribution
class_counts = Counter(labels)
print("Dataset class distribution:", class_counts)

X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models with improvements
models = {
    "SVM": SVC(kernel='linear', class_weight="balanced", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

# Track results for plotting
accuracy_scores = {}
log_losses = {}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc
    
    # Compute log loss for models with probability output
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_proba)
    else:
        loss = None
    log_losses[name] = loss

    print(f"{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, target_names=categories, zero_division=0))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Accuracy plot
plt.figure(figsize=(8, 5))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.show()

# Loss plot
plt.figure(figsize=(8, 5))
loss_models = {k: v for k, v in log_losses.items() if v is not None}
plt.bar(loss_models.keys(), loss_models.values(), color=['blue', 'green', 'red'])
plt.xlabel("Models")
plt.ylabel("Log Loss")
plt.title("Model Log Loss Comparison")
plt.show()

# # Save models, encoder, and scaler
# joblib.dump(label_encoder, "label_encoder.pkl")
# joblib.dump(scaler, "scaler.pkl")
# for name, model in models.items():
#     joblib.dump(model, f"{name}_banana_classifier.pkl")
