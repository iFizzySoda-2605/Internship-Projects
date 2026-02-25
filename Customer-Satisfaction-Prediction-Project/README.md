Customer Support Ticket Analysis and Satisfaction Prediction
This project analyzes customer support ticket data to uncover business insights, perform exploratory analysis, build a predictive model for customer satisfaction, and create a fully interactive dashboard using Streamlit.

📂 Project Structure
Copy
Edit
├── 1_data_preprocessing.py
├── 2_exploratory_data_analysis.py
├── 3_feature_engineering.py
├── 4_model_building.py
├── 5_model_evaluation.py
├── 6_visualization.py
├── cleaned_data.csv
├── requirements.txt
├── README.md
📌 Objective
Analyze customer support tickets to understand behavior and service quality.

Predict customer satisfaction using machine learning models.

Present results through visual insights and an interactive Streamlit dashboard.

🔧 Files Description
1️⃣ 1_data_preprocessing.py
Loads raw dataset customer_support_tickets.csv.

Cleans missing values.

Encodes categorical features using Label Encoding.

Saves the cleaned dataset to cleaned_data.csv.

2️⃣ 2_exploratory_data_analysis.py
Loads cleaned_data.csv.

Maps encoded values to readable labels.

Performs detailed exploratory analysis with 10 visualizations.

Saves all plots (PNG format) in a visuals/ folder:

Correlation Heatmap

Satisfaction Distributions

Ticket Priority and Type

Response and Resolution Time

Ticket Channels and more

3️⃣ 3_feature_engineering.py
Loads and prepares the dataset for modeling.

Drops irrelevant columns (e.g., ticket ID, names).

Splits data into training and test sets.

Applies standardization using StandardScaler.

Saves X_train.csv, X_test.csv, y_train.csv, y_test.csv.

4️⃣ 4_model_building.py
Trains multiple models:

Logistic Regression

Random Forest

Decision Tree

K-Nearest Neighbors

Evaluates accuracy and prints classification reports.

Selects and saves the best model as best_model.pkl.

Also saves the scaler as scaler.pkl.

5️⃣ 5_model_evaluation.py
Loads saved model and scaler.

Evaluates the model on the test dataset.

Prints accuracy, classification report, and confusion matrix.

Saves prediction results to predictions.csv.

Saves confusion matrix plot.

6️⃣ 6_visualization.py
Streamlit dashboard.

Loads cleaned_data.csv and displays:

Project overview and key insights.

12 interactive visualizations (bar charts, pie charts, line plots).

Loaded trained model for satisfaction prediction.

Clean layout with sectioned summaries.

Requires best_model.pkl and cleaned_data.csv.

💡 Technologies Used
Languages: Python

Libraries:

pandas, numpy, matplotlib, seaborn

scikit-learn, joblib, streamlit

Tools:

Jupyter Notebook / PyCharm

GitHub (for version control and deployment)

Dashboard: Streamlit

🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/customer-support-analysis.git
cd customer-support-analysis
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run Streamlit Dashboard:

bash
Copy
Edit
streamlit run 6_visualization.py
📈 Output
Best model with accuracy printed in console.

Predictions saved to predictions.csv.

Visualization dashboard available at local host.

✅ Results & Insights
Email and Phone are the most used channels.

Customers aged 18–45 raised the most tickets.

Satisfaction varies by gender and ticket priority.

The model predicts satisfaction levels with high accuracy.

