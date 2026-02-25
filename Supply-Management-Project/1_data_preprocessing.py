import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("customer_support_tickets.csv")

# Display basic info
print("Initial shape:", df.shape)
print(df.info())

# Handle missing values
df.dropna(inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Save preprocessed data
df.to_csv("cleaned_data.csv", index=False)
print("Data preprocessing complete. Shape:", df.shape)
