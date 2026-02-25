# 📌 Customer Satisfaction Prediction — Machine Learning Analysis
## 📊 Objective

The goal of this project was to analyze customer satisfaction data and evaluate whether Machine Learning can predict satisfaction levels based on ticket-related information like product category, priority, and resolution time.

## 📁 Dataset Overview
1. Feature	Description
2. Ticket ID	Unique Identifier
3. Product Category	Software / Hardware / Wearables
4. Priority	Low / Medium / High
5. Response Time	Time taken to respond (Hours)
6. Resolution Time	Time taken to resolve (Hours)
7. Satisfaction Rating	1–5 rating (67% missing)

### 📌 Only 33% records contained satisfaction ratings → 2,769 usable samples
### 📌 Ratings distributed uniformly (20% each from 1 to 5)

### 🔬 Key Findings & Insights
❌ Data Limitations

* Maximum correlation with satisfaction: 0.029 → almost random
* Response time impact: -0.33 points on satisfaction
* Logistic Regression accuracy = 60.8%
→ Same as predicting all customers are Unsatisfied

* Resolution time range ≤ 25 hours → not enough variance
* No customer-history or issue-content features → no behavioral insights

## 📍 Conclusion:
Dataset appears to be synthetic / test data → Not suitable for predictive ML modeling.

## 🧠 Models Attempted
→ Type	Models Used	Performance Result
1. Binary Classification	Logistic Regression, Random Forest	Accuracy = 60.8% (baseline level)
2. Rating Prediction (Regression)	Linear Regression	R² = -0.0001 → No predictive power

✔️ Preprocessing was correctly done
✔️ Tested multiple algorithms
🔻 No model improved beyond baseline → No learnable pattern

## 📌 Business Insights Extracted

Even though ML didn’t work, we derived valuable product performance insights:

Product Category	Avg. Rating	Feedback
* Software	⭐ 3.09	Best performing
* Hardware	⭐ 2.95	Moderate
* Wearables	⭐ 2.86	🔻 Needs major improvement

### 🔧 Technology Stack
* Python (pandas, numpy, scikit-learn)
* Data Visualization: matplotlib, seaborn
* Jupyter Notebook

### 🚫 Why ML Didn’t Work Here
* Issue	Impact
* 67% missing satisfaction	Very limited training data
* Features contain no predictive signal	ML can’t learn patterns
* Uniformly distributed target values	No natural class separation
* No customer context or issue text	No sentiment / history insights

### 🚀 Recommendations & Future Work
* 1️⃣ Data Collection	Capture satisfaction consistently
* 2️⃣ Add Behavior Insights	Complaint text, customer profile
* 3️⃣ Sentiment Analysis	NLP on support messages
* 4️⃣ Retry ML	After data quality improvements


### 🏁 Final Conclusion
Machine Learning cannot be reliably applied on this dataset in its current form.
However, the analysis provided clear business guidance, especially for improving Wearables product quality and data collection strategy.
