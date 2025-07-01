# 💸 Loan Default Risk Analyzer

A machine learning web app that predicts the probability of a loan applicant defaulting on their loan. Built with XGBoost, Streamlit, and real-world feature engineering, this project simulates how banks evaluate borrower risk.

🔗 **Live App**: [Loan Default Analyzer](https://dhathripenmatsa-loan-default-analyzer-app-u0ji4f.streamlit.app/)  
📂 **Dataset**: Custom CSV-based dataset  
💻 **Model**: XGBoost Classifier + Feature Engineering + SMOTE

---

## 🧠 Features

- Predicts default risk (0–100%) based on financial & personal data
- Custom feature engineering:
  - Loan-to-Income Ratio
  - Debt-to-Income Flag
  - Senior Age Group Flag
  - Short-Term Loan Indicator
- Balanced training using SMOTE
- Tuned XGBoost classifier with ROC-AUC cross-validation
- Deployed via Streamlit Cloud

---

## 📁 Files in This Project

| File                   | Description                               |
|------------------------|-------------------------------------------|
| `app.py`               | Streamlit UI for predictions              |
| `train_model.py`       | ML model training + feature engineering   |
| `encoder.pkl`          | Saved preprocessor (OneHotEncoder)        |
| `loan_default_model.pkl` | Trained XGBoost model                  |
| `requirements.txt`     | Dependencies and pinned sklearn version   |
| `data/Loan_default.csv`| CSV dataset with borrower info            |
| 'predict.py'
---

## 📊 Example Input & Output

**Input:**
- Age: 63
- Income: ₹1,20,000
- Loan: ₹200,000
- Credit Score: 480
- DTI Ratio: 0.95
- Employment: Unemployed  
...

**Output:**
> 🧾 Probability of Default: **87.3%**  
> 🔴 High Risk – likely to default

---

## ✅ Tech Stack

- **Python 3.10+**
- **Scikit-learn 1.6.1**
- **XGBoost**
- **SMOTE (imbalanced-learn)**
- **Streamlit** for web UI
- **Joblib** for saving models
- **GitHub + Streamlit Cloud** for deployment

---

## ✨ Author

[dhathripenmatsa](https://github.com/dhathripenmatsa)
