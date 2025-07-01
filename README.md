# 💸 Loan Default Risk Analyzer

A machine learning web app that predicts the probability of a loan applicant defaulting on their loan. Built with XGBoost, Streamlit, and real-world feature engineering, this project simulates how banks evaluate borrower risk.

🔗 **Demo**: [Loan Default Analyzer](https://dhathripenmatsa-loan-default-analyzer-app-u0ji4f.streamlit.app/)  
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

## 📁 Project Structure

| File / Folder          | Description |
|------------------------|-------------|
| `data/Loan_default.csv`| CSV dataset with training data |
| `app.py`               | Streamlit UI to input user details and predict risk |
| `predict.py`           | Standalone script for making predictions (for testing or API use) |
| `train_model.py`       | Training script for model and encoder with feature engineering |
| `encoder.pkl`          | Saved OneHotEncoder from training pipeline |
| `loan_default_model.pkl` | Trained XGBoost model |
| `requirements.txt`     | All Python dependencies with version pins |
| `README.md`            | You are reading it :) |

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
