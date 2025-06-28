# 🏦 Loan Default Risk Analyzer

This is a machine learning web application that predicts the probability of a loan applicant **defaulting**. It uses a **Gradient Boosting Classifier** trained on real-world financial attributes, and is deployed using **Streamlit**.

🔗 **Live Demo**: [Click here to try the app](https://dhathripenmatsa-loan-default-analyzer-app-u0ji4f.streamlit.app/)

---

## 🚀 Features

- 📊 Predicts **risk of default** based on financial & demographic inputs
- 🧠 Uses **scikit-learn ML model** (Gradient Boosting Classifier)
- 🌐 Built with **Streamlit** for fast, interactive web UI
- 🔐 Inputs include: Credit Score, Income, Loan Amount, DTI Ratio, Employment Type, Co-Signer, etc.
- 📝 Easily customizable & extendable (e.g. SHAP explanations, model retraining)

---

## 📊 Sample Prediction

> Enter details of an applicant like age, income, credit score, etc.  
> The app returns a probability and classifies them as **High-risk** or **Low-risk**.

![App Screenshot](https://i.imgur.com/Ow3RW7a.png)

---

## 🧠 How It Works

1. Load and preprocess `Loan_default.csv`
2. Train a model and save it as `loan_default_model.pkl`
3. Streamlit app loads the model and takes user input
4. Input is preprocessed → prediction made → risk displayed

---

## 🗂 Folder Structure

loan-default-risk/
├── data/
│ └── Loan_default.csv # training data
├── train_model.py # model training script
├── predict.py # manual prediction script
├── app.py # Streamlit web app
├── loan_default_model.pkl # saved ML model
├── requirements.txt # dependency file
└── README.md # you're reading it

---

## 📦 Tech Stack

Python 3
pandas
scikit-learn
joblib
Streamlit

## 🙋‍♀️ Author
Dhathri Penmatsa
