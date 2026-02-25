# 4_model_building.py

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load training and testing data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary of models to try
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

best_model_name = None
best_accuracy = 0
best_model = None

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    print(f"✅ Accuracy of {name}: {acc:.4f}")
    print("📊 Classification Report:")
    print(classification_report(y_test, predictions))
    print("🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Save the best model
if best_model:
    joblib.dump(best_model, "best_model.pkl")
    print(f"\n✅ Best model saved: {best_model_name} with accuracy {best_accuracy:.4f}")
else:
    print("\n❌ No model could be trained.")

# Save the scaler as well
joblib.dump(scaler, "scaler.pkl")
