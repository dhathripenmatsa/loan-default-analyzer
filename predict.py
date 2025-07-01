"""
predict.py
==========

Quick terminal script to test the trained model on a single applicant.
Edit `sample` dict as needed.
"""

import joblib, pandas as pd

# Load artefacts
model   = joblib.load("loan_default_model.pkl")
encoder = joblib.load("encoder.pkl")

# --- Sample applicant (edit freely) --------------------------------
sample = {
    "Age": 70,
    "Income": 100000,
    "LoanAmount": 200000,
    "CreditScore": 350,
    "MonthsEmployed": 4,
    "NumCreditLines": 2,
    "InterestRate": 24.5,
    "LoanTerm": 12,
    "DTIRatio": 0.98,
    "Education": "High School",
    "EmploymentType": "Unemployed",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "Yes",
    "LoanPurpose": "Debt Consolidation",
    "HasCoSigner": "No",
}

# --- Engineered features (match training script) -------------------
sample["Loan_to_Income_Ratio"] = sample["LoanAmount"] / (sample["Income"] + 1)
sample["Employed"]             = int(sample["EmploymentType"] != "Unemployed")
sample["ShortLoanTerm"]        = int(sample["LoanTerm"] <= 12)
sample["High_DTI"]             = int(sample["DTIRatio"] >= 0.45)
sample["Senior"]               = int(sample["Age"] >= 60)

# Convert to DataFrame
df_input = pd.DataFrame([sample])
print("🚨 EXPECTED INPUT COLUMNS:", encoder.feature_names_in_)
print("🧾 YOUR INPUT COLUMNS     :", df_input.columns.tolist())

# Transform & predict
X_enc = encoder.transform(df_input)
prob  = model.predict_proba(X_enc)[0][1]

print(f"\n🧾 Probability of Default: {prob*100:.2f}%")
print("Risk Category:",
      "🔴 High"   if prob >= 0.7 else
      "🟠 Medium" if prob >= 0.4 else
      "🟢 Low")

