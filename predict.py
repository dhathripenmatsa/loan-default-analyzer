import joblib
import pandas as pd

# 1. Load the saved pipeline (prep + model)
model = joblib.load("loan_default_model.pkl")

# 2. Enter a sample applicant (replace values as needed)
sample = {
    "Age": 32,
    "Income": 55000,
    "LoanAmount": 18000,
    "CreditScore": 710,
    "MonthsEmployed": 60,
    "NumCreditLines": 4,
    "InterestRate": 10.5,
    "LoanTerm": 36,
    "DTIRatio": 0.24,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Single",
    "HasMortgage": "No",
    "HasDependents": "No",
    "LoanPurpose": "Car",
    "HasCoSigner": "No"
}

input_df = pd.DataFrame([sample])

# 3. Predict
prob_default = model.predict_proba(input_df)[0][1]
label        = model.predict(input_df)[0]

print(f"Probability of default: {prob_default:.1%}")
print("Prediction:", "❌ High‑risk borrower" if label else "✅ Low‑risk borrower")
