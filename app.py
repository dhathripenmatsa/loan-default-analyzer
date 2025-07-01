"""
app.py
======

Streamlit web UI for the Loan Default Risk Analyzer.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Load artefacts
model   = joblib.load("loan_default_model.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(page_title="Loan Default Risk Analyzer", layout="centered")
st.title("💸 Loan Default Risk Analyzer")
st.write("Fill in the applicant details and click **Predict** to estimate default risk.")

# ------------------------------------------------------------------
# UI form
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        age          = st.slider("Age", 18, 100, 35)
        income       = st.number_input("Annual Income (₹)", 0, 10_000_000, 300_000, step=10_000)
        loan_amount  = st.number_input("Loan Amount (₹)",   1_000, 5_000_000, 200_000, step=10_000)
        credit_score = st.slider("Credit Score", 300, 900, 650)
        interest     = st.slider("Interest Rate (%)", 0.0, 50.0, 15.0)

    with col2:
        months_emp   = st.number_input("Months Employed", 0, 600, 24)
        lines        = st.number_input("Number of Credit Lines", 1, 20, 3)
        loan_term    = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
        dti_ratio    = st.slider("Debt‑to‑Income Ratio", 0.0, 2.0, 0.4)

    education       = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self‑Employed", "Unemployed"])
    marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    has_mortgage    = st.radio("Has Mortgage?", ["Yes", "No"])
    has_dependents  = st.radio("Has Dependents?", ["Yes", "No"])
    purpose         = st.selectbox("Loan Purpose", ["Debt Consolidation", "Education", "Medical", "Home", "Business"])
    has_cosigner    = st.radio("Has Co‑Signer?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

# ------------------------------------------------------------------
# Prediction
if submitted:
    data = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_emp,
        "NumCreditLines": lines,
        "InterestRate": interest,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": purpose,
        "HasCoSigner": has_cosigner,
        # Engineered features
        "Loan_to_Income_Ratio": loan_amount / (income + 1),
        "Employed": int(employment_type != "Unemployed"),
        "ShortLoanTerm": int(loan_term <= 12),
        "High_DTI": int(dti_ratio >= 0.45),
        "Senior": int(age >= 60),
    }

    df_input = pd.DataFrame([data])

    # Ensure columns match encoder
    missing = set(encoder.feature_names_in_) - set(df_input.columns)
    if missing:
        st.error(f"Missing columns for encoder: {missing}")
    else:
        X_enc = encoder.transform(df_input)
        prob  = model.predict_proba(X_enc)[0][1]

        st.markdown(f"### 🧾 Probability of Default: **{prob*100:.2f}%**")
        if prob >= 0.7:
            st.error("🔴 High Risk – likely to default")
        elif prob >= 0.4:
            st.warning("🟠 Medium Risk – caution advised")
        else:
            st.success("🟢 Low Risk – likely to repay")


