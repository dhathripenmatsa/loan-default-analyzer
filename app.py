# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoder
model = joblib.load("loan_default_model.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(page_title="Loan Default Risk Analyzer", layout="centered")
st.title("💸 Loan Default Risk Analyzer")
st.write("Enter applicant details below to estimate the probability of loan default.")

# Input form
with st.form("loan_form"):
    age = st.slider("Age", 18, 100, 30)
    income = st.number_input("Annual Income (₹)", min_value=0, value=300000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, value=150000)
    credit_score = st.slider("Credit Score", 300, 900, 650)
    months_employed = st.number_input("Months Employed", min_value=0, value=24)
    credit_lines = st.number_input("Number of Credit Lines", min_value=1, value=2)
    interest_rate = st.slider("Interest Rate (%)", 0.0, 50.0, 12.0)
    loan_term = st.number_input("Loan Term (months)", min_value=1, value=36)
    dti_ratio = st.slider("Debt-To-Income Ratio", 0.0, 2.0, 0.5)

    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    has_mortgage = st.radio("Has Mortgage?", ["Yes", "No"])
    has_dependents = st.radio("Has Dependents?", ["Yes", "No"])
    loan_purpose = st.selectbox("Loan Purpose", ["Debt Consolidation", "Education", "Medical", "Home", "Business"])
    has_cosigner = st.radio("Has Co‑Signer?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Default Risk")

if submitted:
    # Create DataFrame
    input_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
        
        # Engineered features (same as train_model.py)
        "Loan_to_Income_Ratio": loan_amount / (income + 1),
        "Employed": int(employment_type != "Unemployed"),
        "ShortLoan": int(loan_term <= 12),
        "High_DTI": int(dti_ratio >= 0.45),
        "Senior": int(age >= 60),
    }])

    # Apply encoder to match training features
    X_encoded = encoder.transform(input_data)

    # Predict
    prob = model.predict_proba(X_encoded)[0][1]
    st.success(f"🧾 Probability of Default: **{round(prob * 100, 2)}%**")

    # Risk Interpretation
    if prob >= 0.7:
        st.error("🔴 High Risk – likely to default")
    elif prob >= 0.4:
        st.warning("🟠 Medium Risk – caution advised")
    else:
        st.info("🟢 Low Risk – likely to repay")

