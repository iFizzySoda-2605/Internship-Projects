import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")

# Rename columns for consistency (same as in EDA)
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
      .str.replace("-", "_")
)

# Drop columns not useful for modeling
drop_cols = [
    "ticket_id",
    "customer_name",
    "customer_email",
    "ticket_subject",
    "ticket_description",
    "resolution",  # Often similar to target
]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Define features (X) and target (y)
X = df.drop(columns=["customer_satisfaction_rating"], errors="ignore")
y = df["customer_satisfaction_rating"]

# Optional: if y is continuous, you can convert it into classes (e.g., Good/Bad)
# For now, we keep it as regression target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save datasets
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ Feature engineering complete. Files saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
