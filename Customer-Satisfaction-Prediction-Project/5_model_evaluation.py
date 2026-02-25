# 5_model_evaluation.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# Load saved model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Predict
predictions = model.predict(X_test_scaled)

# Evaluate
acc = accuracy_score(y_test, predictions)
print(f" Model Accuracy on Test Set: {acc:.4f}")
print("\n Classification Report:")
print(classification_report(y_test, predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Save predictions
results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})
results_df.to_csv("predictions.csv", index=False)
print(" Predictions saved to predictions.csv")
