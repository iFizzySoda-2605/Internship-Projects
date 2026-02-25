import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the cleaned data
df = pd.read_csv("cleaned_data.csv")

# 2. Normalize column names for convenience
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
      .str.replace("-", "_")
)

# 3. (Optional) Map numeric codes to human‑readable labels
channel_map = {1: "Email", 2: "Phone", 3: "Chat", 4: "Web"}
type_map    = {1: "Technical", 2: "Billing", 3: "Inquiry", 4: "Feedback"}

if "ticket_channel" in df:
    df["ticket_channel_label"] = df["ticket_channel"].map(channel_map)
if "ticket_type" in df:
    df["ticket_type_label"]    = df["ticket_type"].map(type_map)

# Ensure visuals directory exists
os.makedirs("visuals", exist_ok=True)

# Set a consistent Seaborn style
sns.set(style="whitegrid")

# 1. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("visuals/correlation_heatmap.png")
plt.close()

# 2. Customer Satisfaction Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="customer_satisfaction_rating", data=df, palette="Set2")
plt.title("Customer Satisfaction Distribution")
plt.xlabel("Satisfaction Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("visuals/customer_satisfaction_distribution.png")
plt.close()

# 3. First Response Time by Priority
plt.figure(figsize=(8, 6))
sns.boxplot(x="ticket_priority", y="first_response_time", data=df, palette="Pastel1")
plt.title("First Response Time by Ticket Priority")
plt.xlabel("Ticket Priority")
plt.ylabel("First Response Time")
plt.tight_layout()
plt.savefig("visuals/first_response_by_priority.png")
plt.close()

# 4. Resolution Time by Priority
plt.figure(figsize=(8, 6))
sns.boxplot(x="ticket_priority", y="time_to_resolution", data=df, palette="Pastel2")
plt.title("Resolution Time by Ticket Priority")
plt.xlabel("Ticket Priority")
plt.ylabel("Time to Resolution")
plt.tight_layout()
plt.savefig("visuals/resolution_by_priority.png")
plt.close()

# 5. Ticket Channel Distribution
if "ticket_channel_label" in df:
    plt.figure(figsize=(7, 5))
    sns.countplot(x="ticket_channel_label", data=df, palette="Set3")
    plt.title("Ticket Channel Distribution")
    plt.xlabel("Channel")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visuals/ticket_channel_distribution.png")
    plt.close()

# 6. Ticket Type Distribution
if "ticket_type_label" in df:
    plt.figure(figsize=(7, 5))
    sns.countplot(x="ticket_type_label", data=df, palette="Set1")
    plt.title("Ticket Type Distribution")
    plt.xlabel("Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visuals/ticket_type_distribution.png")
    plt.close()

# 7. Time to Resolution Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["time_to_resolution"], bins=30, kde=True, color="skyblue")
plt.title("Time to Resolution Distribution")
plt.xlabel("Time to Resolution")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("visuals/time_to_resolution_distribution.png")
plt.close()

# 8. Satisfaction by Ticket Type
if "ticket_type_label" in df:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="ticket_type_label", y="customer_satisfaction_rating", data=df, palette="coolwarm")
    plt.title("Satisfaction by Ticket Type")
    plt.xlabel("Ticket Type")
    plt.ylabel("Satisfaction Rating")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visuals/satisfaction_by_ticket_type.png")
    plt.close()

# 9. Satisfaction by Channel
if "ticket_channel_label" in df:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="ticket_channel_label", y="customer_satisfaction_rating", data=df, palette="viridis")
    plt.title("Satisfaction by Ticket Channel")
    plt.xlabel("Ticket Channel")
    plt.ylabel("Satisfaction Rating")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visuals/satisfaction_by_channel.png")
    plt.close()

# 10. Satisfaction by Priority
plt.figure(figsize=(7, 5))
sns.boxplot(x="ticket_priority", y="customer_satisfaction_rating", data=df, palette="rocket")
plt.title("Satisfaction by Ticket Priority")
plt.xlabel("Ticket Priority")
plt.ylabel("Satisfaction Rating")
plt.tight_layout()
plt.savefig("visuals/satisfaction_by_priority.png")
plt.close()

print("✅ All 10 EDA visuals saved in the 'visuals/' folder.")
