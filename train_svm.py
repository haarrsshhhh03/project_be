import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# Load dataset
df = pd.read_csv("hand_gestures.csv")

# Split labels and features
X = df.iloc[:, 1:].values  # All columns except 'label'
y = df.iloc[:, 0].values   # Only the 'label' column

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train SVM model
#svm_model = SVC(kernel='linear', probability=True, random_state=10)  # Using linear kernel

#svm_model= SVC(kernel='rbf', C=1.0, gamma='scale') # Using rbf

svm_model = SVC(kernel='poly', degree=3, C=1.0) # Using POly

svm_model.fit(X_train, y_train)

# Test accuracy
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(svm_model, "svm_model.pkl")
print("SVM Model saved as svm_model.pkl")

# Confusion Matrix
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I",  "K", "L", "M", 
                "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y"]

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - SVM")
plt.show()

# Compute F1 Score
f1_train = f1_score(y_train, svm_model.predict(X_train), average='weighted')
f1_test = f1_score(y_test, y_pred, average='weighted')

print(f"F1 Score (Train): {f1_train:.4f}")
print(f"F1 Score (Test): {f1_test:.4f}")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_labels))
