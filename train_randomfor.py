import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# Load the dataset
df = pd.read_csv("hand_gestures.csv")

#  Step 1: Feature and Label Split

X = df.iloc[:, 1:].values  # Features (XYZ landmarks)
y = df.iloc[:, 0].values   # Labels (gesture classes)

#  Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Model Initialization
model = RandomForestClassifier(
    n_estimators=200,        # Number of trees in forest
    max_depth=None,            # Limit depth to avoid overfitting
    min_samples_split=4,     # Minimum samples to split node
    random_state=42
)

# ================================
# 🏋️ Step 4: Train the Model
# ================================
model.fit(X_train, y_train)

# ================================
# 🔍 Step 5: Evaluate on Test Set
# ================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# Save model for inference
joblib.dump(model, "model.pkl")
print("💾 Model saved as model.pkl")

# ================================
# 📊 Step 6: Confusion Matrix Plot
# ================================
class_labels = sorted(list(set(y)))  # Auto-detect classes from data

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("🧩 Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()

# ================================
# 📈 Step 7: F1 Score Evaluation
# ================================
f1_train = f1_score(y_train, model.predict(X_train), average='weighted')
f1_test = f1_score(y_test, y_pred, average='weighted')
print(f"🏋️ F1 Score (Train): {f1_train:.4f}")
print(f"🧪 F1 Score (Test): {f1_test:.4f}")

# Optional: Plot F1 score bar
plt.figure(figsize=(6,4))
plt.bar(['Train F1', 'Test F1'], [f1_train, f1_test], color=['green', 'orange'])
plt.title("🎯 F1 Score Comparison (Train vs Test)")
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()

# ================================
# 📄 Step 8: Classification Report
# ================================
print("\n🧾 Classification Report:\n")
print(classification_report(y_test, y_pred))
