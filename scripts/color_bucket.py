import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Function to extract color histogram features
def extract_color_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # Resize to a fixed size
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Compute histogram for Hue channel
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Compute mean and std deviation of HSV channels
    mean_hue = np.mean(hsv[:, :, 0])
    std_hue = np.std(hsv[:, :, 0])
    mean_sat = np.mean(hsv[:, :, 1])
    std_sat = np.std(hsv[:, :, 1])
    
    return np.hstack([hist, mean_hue, std_hue, mean_sat, std_sat])

# Load dataset
categories = ['Ripe', 'Overripe', 'Green', 'Decay']
data_dir = "AUGMENTED_DATA"
X, y = [], []

for category in categories:
    path = os.path.join(data_dir, category)
    for img_path in glob.glob(os.path.join(path, "*.jpg")):
        features = extract_color_histogram(img_path)
        X.append(features)
        y.append(category)

X = np.array(X)
y = np.array(y)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train & Evaluate Models
models = {
    'SVM': SVC(kernel='linear', probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {model.score(X_test, y_test):.2f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
    plt.xlabel(" Predicted ")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

