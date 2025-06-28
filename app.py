import joblib, pandas as pd, streamlit as st

model = joblib.load("loan_default_model.pkl")
st.title("🏦 Loan Default Risk Analyzer")

# --- numeric inputs
age = st.number_input("Age", 18, 100, 32)
income = st.number_input("Annual Income", 0, 1_000_000, 55000)
loan_amt = st.number_input("Loan Amount", 0, 500_000, 18000)
credit = st.number_input("Credit Score", 300, 850, 710)
months_emp = st.number_input("Months Employed", 0, 600, 60)
num_lines = st.number_input("Number of Credit Lines", 0, 20, 4)
int_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 10.5)
term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
dti = st.number_input("DTI Ratio", 0.0, 1.0, 0.24)

# --- categorical inputs
education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "Other"])
emp_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital = st.selectbox("Marital Status", ["Single", "Married"])
has_mort = st.radio("Has Mortgage?", ["Yes", "No"])
has_dep = st.radio("Has Dependents?", ["Yes", "No"])
purpose = st.selectbox("Loan Purpose", ["Car", "Home Improvement", "Debt Consolidation", "Other"])
cosign = st.radio("Has Co‑Signer?", ["Yes", "No"])

if st.button("Predict Default Risk"):
    sample = pd.DataFrame([{
        "Age": age, "Income": income, "LoanAmount": loan_amt,
        "CreditScore": credit, "MonthsEmployed": months_emp,
        "NumCreditLines": num_lines, "InterestRate": int_rate,
        "LoanTerm": term, "DTIRatio": dti, "Education": education,
        "EmploymentType": emp_type, "MaritalStatus": marital,
        "HasMortgage": has_mort, "HasDependents": has_dep,
        "LoanPurpose": purpose, "HasCoSigner": cosign
    }])
    proba = model.predict_proba(sample)[0][1]
    pred  = model.predict(sample)[0]
    st.write(f"**Probability of default:** {proba:.1%}")
    if pred:
        st.error("❌ High‑risk borrower (likely to default)")
    else:
        st.success("✅ Low‑risk borrower (likely to repay)")
