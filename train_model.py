"""
train_model.py
==============

• Loads data/Loan_default.csv
• Adds engineered features (Loan_to_Income_Ratio, Employed, ShortLoanTerm, High_DTI, Senior)
• One‑hot‑encodes categoricals
• Balances training set with SMOTE
• Trains tuned XGBoost
• Saves:
    ├─ loan_default_model.pkl   (model)
    └─ encoder.pkl              (pre‑processor)
"""

import pathlib, joblib, warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 1. Load data
DATA_PATH = pathlib.Path("data") / "Loan_default.csv"
df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------------
# 2. Basic cleaning + feature engineering (keep these names!)
df.drop(columns=["LoanID"], inplace=True)

df["Loan_to_Income_Ratio"] = df["LoanAmount"] / (df["Income"] + 1)
df["Employed"]             = (df["EmploymentType"] != "Unemployed").astype(int)
df["ShortLoanTerm"]        = (df["LoanTerm"] <= 12).astype(int)
df["High_DTI"]             = (df["DTIRatio"] >= 0.45).astype(int)
df["Senior"]               = (df["Age"]       >= 60).astype(int)

X = df.drop("Default", axis=1)
y = df["Default"]

# ------------------------------------------------------------------
# 3. Encode categoricals
cat_cols = X.select_dtypes(include="object").columns
encoder  = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough"
)

X_encoded = encoder.fit_transform(X)

# ------------------------------------------------------------------
# 4. Train/test split + SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)

# ------------------------------------------------------------------
# 5. XGBoost model (tuned starter params)
pos, neg = y_train_sm.sum(), len(y_train_sm) - y_train_sm.sum()
scale_pos_weight = neg / pos

model = XGBClassifier(
    random_state=42,
    eval_metric="auc",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=1,
    scale_pos_weight=scale_pos_weight
)
model.fit(X_train_sm, y_train_sm)

# ------------------------------------------------------------------
# 6. Evaluation
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("Accuracy :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("ROC‑AUC  :", round(roc_auc_score(y_test, y_proba), 3))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Paid", "Default"]).plot()
plt.tight_layout()
plt.show()

# 5‑fold CV for robustness
cv_auc = cross_val_score(model, X_encoded, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="roc_auc")
print("Cross‑val ROC‑AUC:", cv_auc.mean().round(3), "±", cv_auc.std().round(3))

# ------------------------------------------------------------------
# 7. Save model & encoder
joblib.dump(model,   "loan_default_model.pkl")
joblib.dump(encoder, "encoder.pkl")
print("\n✅  Saved model ➜ loan_default_model.pkl")
print("✅  Saved encoder ➜ encoder.pkl")



