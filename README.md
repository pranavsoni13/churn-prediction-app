🚀 Customer Churn Prediction App

A machine learning web application that predicts whether a customer is likely to churn or not. Built using a Random Forest model and deployed with Streamlit.

🔗 **Live App:** https://churn-prediction-app-pranav132004.streamlit.app

---

📌 Overview

Customer churn is a major problem in industries like telecom and SaaS. This project uses machine learning to identify customers who are likely to leave, allowing businesses to take preventive action.

---

🧠 Features

* Predicts customer churn (Yes/No)
* Shows churn probability
* Interactive web interface using Streamlit
* End-to-end ML pipeline (data → model → deployment)

---

## ⚙️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit Cloud
* **Version Control:** Git & GitHub

---

🔄 ML Workflow

1. **Data Preprocessing**

   * Handled missing values
   * Encoded categorical variables
   * Feature transformation

2. **Exploratory Data Analysis (EDA)**

   * Churn distribution analysis
   * Feature relationships (tenure, contract, charges)

3. **Model Training**

   * Logistic Regression (baseline)
   * Random Forest (final model)

4. **Model Optimization**

   * Handled class imbalance
   * Applied threshold tuning (0.5 → 0.3)
   * Improved churn recall from ~47% to ~70%

5. **Deployment**

   * Built interactive UI using Streamlit
   * Deployed on Streamlit Cloud

---

## 📊 Key Insights

* Customers with **low tenure** are more likely to churn
* **Month-to-month contracts** have higher churn rates
* Higher **monthly charges** increase churn probability
* Additional services (security, support) reduce churn

---

🧪 How to Run Locally

```bash
git clone https://github.com/pranavsoni13/churn-prediction-app.git
cd churn-prediction-app
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 📁 Project Structure

```
churn_project/
│
├── app/                # Streamlit app
├── src/                # Preprocessing & training scripts
├── models/             # Saved ML model
├── notebooks/          # Jupyter notebook (EDA + experiments)
├── data/               # Dataset
└── requirements.txt
```

---

## 🎯 Future Improvements

* Add more models (XGBoost, SVM)
* Improve UI/UX
* Add user authentication
* Deploy using Docker or cloud platforms

---

## 🙌 Author

**Pranav Soni**
AI & Data Science Student

---

⭐ If you like this project, consider giving it a star!
