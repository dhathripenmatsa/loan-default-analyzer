# predict.py

import joblib
import pandas as pd

# Load model and encoder
model = joblib.load("loan_default_model.pkl")
encoder = joblib.load("encoder.pkl")

# Exact input with required columns and expected values
input_dict = {
    "Age": 70,
    "Income": 100000,
    "LoanAmount": 200000,
    "CreditScore": 350,
    "MonthsEmployed": 4,
    "NumCreditLines": 2,
    "InterestRate": 24.5,
    "LoanTerm": 12,
    "DTIRatio": 0.98,
    "Education": "High School",               # Must match training categories
    "EmploymentType": "Unemployed",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "Yes",
    "LoanPurpose": "Debt Consolidation",
    "HasCoSigner": "No",
}

# predict.py  (only the engineered‑feature block changes)

input_dict["Loan_to_Income_Ratio"] = input_dict["LoanAmount"] / (input_dict["Income"] + 1)
input_dict["Employed"]             = 0 if input_dict["EmploymentType"] == "Unemployed" else 1
input_dict["ShortLoanTerm"]        = int(input_dict["LoanTerm"] <= 12)   # <-- use this name
input_dict["High_DTI"]             = int(input_dict["DTIRatio"] >= 0.45)
input_dict["Senior"]               = int(input_dict["Age"] >= 60)

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Transform input using encoder
X_encoded = encoder.transform(input_df)

# Predict
prob = model.predict_proba(X_encoded)[0][1]
print(f"🧾 Probability of Default: {round(prob * 100, 2)}%")

# Risk interpretation
if prob >= 0.7:
    print("🔴 High Risk – likely to default")
elif prob >= 0.4:
    print("🟠 Medium Risk – caution advised")
else:
    print("🟢 Low Risk – likely to repay")

