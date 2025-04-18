import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# dataset
df = pd.read_csv("hand_gestures.csv")

# Split labels and features
X = df.iloc[:, 1:].values  # All columns except 'label'
y = df.iloc[:, 0].values   # Only the 'label' column

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=30, random_state=10)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

#saving 
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")



#Confusion matrix
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I",  "K", "L", "M", 
                "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y"]

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()




# Compute F1 Score
f1_train = f1_score(y_train, model.predict(X_train), average='weighted')
f1_test = f1_score(y_test, y_pred, average='weighted')



# Classification Report
print("Classification :\n", classification_report(y_test, y_pred))
