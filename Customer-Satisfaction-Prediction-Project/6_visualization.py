import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Customer Support Ticket Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")
    return df

df = load_data()

# Title
st.title("📊 Customer Support Ticket Analysis Dashboard")

# Project Goal
st.header("🎯 Project Goal")
st.write("""
The objective of this project is to analyze customer support ticket data and uncover key business insights related to ticket trends, customer satisfaction, and product engagement. This analysis also involves predictive modeling to forecast customer satisfaction based on ticket and demographic data.
""")

# Dataset Overview
st.header("📁 Dataset Overview")
st.write("Here's a quick look at the dataset used in this project:")
st.dataframe(df.head())

# Dataset Description
st.markdown("### 🧾 Columns include:")
st.markdown("""
- `Ticket ID`, `Customer Name`, `Customer Email`, `Customer Age`, `Customer Gender`, `Product Purchased`, `Date of Purchase`,  
  `Ticket Type`, `Ticket Subject`, `Ticket Description`, `Ticket Status`, `Resolution`, `Ticket Priority`, `Ticket Channel`,  
  `First Response Time`, `Time to Resolution`, `Customer Satisfaction Rating`
""")

# Key Insights
st.header("🔑 Key Insights")
st.markdown("""
- Most tickets were received via **Email** and **Phone**.
- **Females** gave higher average satisfaction ratings than males.
- The most common ticket types were **Technical Issues** and **Cancellations**.
- A few key products account for the majority of purchases.
""")

# Visualizations
st.header("📊 Data Visualizations")

# 1. Tickets Raised by Age Group
st.subheader("1. Tickets Raised by Age Group")
col1, col2 = st.columns(2)
with col1:
    df['Age_Group'] = pd.cut(df['Customer Age'], bins=[0, 18, 30, 45, 60, 100], labels=['<18', '18-30', '30-45', '45-60', '60+'])
    age_counts = df.groupby('Age_Group')['Ticket ID'].count()
    fig, ax = plt.subplots(figsize=(4, 2.5))
    sns.barplot(x=age_counts.index, y=age_counts.values, palette='pastel', ax=ax)
    ax.set_ylabel("Tickets")
    st.pyplot(fig)
with col2:
    st.markdown("This chart shows how many tickets were raised by each age group, helping identify which demographics need the most support.")

# 2. Ticket Priority Distribution
st.subheader("2. Ticket Priority Distribution")
col1, col2 = st.columns(2)
with col1:
    priority_counts = df['Ticket Priority'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
    ax1.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)
with col2:
    st.markdown("Pie chart showing support ticket spread across priority levels (High, Medium, Low).")

# 3. Ticket Type Distribution
st.subheader("3. Ticket Type Distribution")
col1, col2 = st.columns(2)
with col1:
    fig2, ax2 = plt.subplots(figsize=(4.5, 2.5))
    sns.countplot(data=df, x='Ticket Type', order=df['Ticket Type'].value_counts().index, palette='muted', ax=ax2)
    plt.xticks(rotation=30)
    st.pyplot(fig2)
with col2:
    st.markdown("Shows the volume of each type of support request like Technical Issue, Cancellation, etc.")

# 4. Top 10 Products Purchased
st.subheader("4. Top 10 Products Purchased")
col1, col2 = st.columns(2)
with col1:
    top_products = df['Product Purchased'].value_counts().nlargest(10)
    fig3, ax3 = plt.subplots(figsize=(4.5, 2.5))
    sns.barplot(x=top_products.index, y=top_products.values, palette='Blues_d', ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)
with col2:
    st.markdown("This bar chart identifies the most frequently purchased products that might generate more support issues.")

# 5. Top Items Purchased by Gender
st.subheader("5. Top Items Purchased by Gender")
gender_product = df.groupby(['Customer Gender', 'Product Purchased']).size().reset_index(name='Count')
top_gender_product = gender_product.sort_values(by='Count', ascending=False).groupby('Customer Gender').head(5)
fig4, ax4 = plt.subplots(figsize=(6, 3))
sns.barplot(data=top_gender_product, x='Product Purchased', y='Count', hue='Customer Gender', ax=ax4)
plt.xticks(rotation=45)
st.pyplot(fig4)
st.markdown("Shows top 5 products by gender to analyze customer interests and issues by demographic.")

# 6. Average Satisfaction by Gender
st.subheader("6. Average Satisfaction by Gender")
col1, col2 = st.columns(2)
with col1:
    avg_sat = df.groupby('Customer Gender')['Customer Satisfaction Rating'].mean()
    fig5, ax5 = plt.subplots(figsize=(4, 2.5))
    sns.barplot(x=avg_sat.index, y=avg_sat.values, palette='coolwarm', ax=ax5)
    st.pyplot(fig5)
with col2:
    st.markdown("Displays how satisfaction varies between genders.")

# 7. Ticket Channel Distribution
st.subheader("7. Ticket Channel Distribution")
col1, col2 = st.columns(2)
with col1:
    fig6, ax6 = plt.subplots(figsize=(4.5, 2.5))
    sns.countplot(data=df, x='Ticket Channel', palette='Set2', order=df['Ticket Channel'].value_counts().index, ax=ax6)
    st.pyplot(fig6)
with col2:
    st.markdown("Shows which channels (email, phone, social media, etc.) customers prefer to raise support tickets.")

# 8. Customer Gender Distribution
st.subheader("8. Customer Gender Distribution")
col1, col2 = st.columns(2)
with col1:
    fig7, ax7 = plt.subplots(figsize=(3.5, 2.5))
    sns.countplot(data=df, x='Customer Gender', palette='Accent', ax=ax7)
    st.pyplot(fig7)
with col2:
    st.markdown("Displays the overall gender distribution of the customer base.")

# 9. Ticket Status Distribution
st.subheader("9. Ticket Status Distribution")
col1, col2 = st.columns(2)
with col1:
    fig8, ax8 = plt.subplots(figsize=(4.5, 2.5))
    sns.countplot(data=df, x='Ticket Status', order=df['Ticket Status'].value_counts().index, palette='husl', ax=ax8)
    st.pyplot(fig8)
with col2:
    st.markdown("Indicates how many tickets are open, resolved, or pending.")

# 10. Customer Satisfaction Distribution
st.subheader("10. Customer Satisfaction Distribution")
fig9, ax9 = plt.subplots(figsize=(6, 3))
sns.histplot(df['Customer Satisfaction Rating'], bins=10, kde=True, ax=ax9)
st.pyplot(fig9)
st.markdown("Histogram showing distribution of satisfaction scores across all customers.")

# 11. Tickets Raised Over Time
st.subheader("11. Tickets Raised Over Time")
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'])
tickets_per_day = df.groupby('Date of Purchase')['Ticket ID'].count()
fig10, ax10 = plt.subplots(figsize=(6, 3))
tickets_per_day.plot(ax=ax10)
ax10.set_ylabel("Tickets")
ax10.set_xlabel("Date")
st.pyplot(fig10)
st.markdown("This line chart shows support volume over time, useful for spotting peaks related to product launches or issues.")

# 12. Top 10 Ticket Inquiries
st.subheader("12. Top 10 Ticket Inquiries")
if 'Ticket Subject' in df.columns:
    top_inquiries = df['Ticket Subject'].value_counts().nlargest(10)
    fig11, ax11 = plt.subplots(figsize=(6, 3))
    sns.barplot(x=top_inquiries.values, y=top_inquiries.index, palette='viridis', ax=ax11)
    st.pyplot(fig11)
    st.markdown("These are the most frequently raised topics, helping identify common concerns.")
else:
    st.warning(" 'Ticket Subject' column not found in dataset.")

# Model Summary
st.header("🧠 Model Summary")
try:
    model = joblib.load("best_model.pkl")
    st.success("✔ Model loaded successfully!")
    st.markdown("We used a **Decision Tree Classifier** trained to predict customer satisfaction based on features such as ticket type, priority, and demographics.")
    st.markdown("Model performance metrics can be displayed here.")
except:
    st.error("⚠ Could not load the model. Please check the file path.")

# Conclusion
st.header("📌 Conclusion")
st.markdown("""
- Most support tickets were generated by young and middle-aged customers.
- Email is the dominant channel for raising tickets.
- Customer satisfaction varies across genders, with females reporting higher ratings.
- Key product and ticket type trends can guide business strategy and service improvements.
- Machine learning adds value by forecasting customer satisfaction levels and identifying service bottlenecks.
""")

# Footer
st.markdown("---")
st.markdown("© 2025 Customer Support Ticket Analysis | Powered by Streamlit")
