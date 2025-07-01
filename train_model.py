# train_model.py — Improved Version (Phase 1 Enhancements)

import pathlib, joblib, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# 1. Load data --------------------------------------------------
DATA_PATH = pathlib.Path("data") / "Loan_default.csv"
df = pd.read_csv(DATA_PATH)

# 2. Basic cleaning + Feature Engineering -----------------------
df.drop(columns=["LoanID"], inplace=True)

# New engineered features
df["Loan_to_Income_Ratio"] = df["LoanAmount"] / (df["Income"] + 1)
df["Employed"] = (df["EmploymentType"] != "Unemployed").astype(int)
df["ShortLoan"] = (df["LoanTerm"] <= 12).astype(int)
df["High_DTI"] = (df["DTIRatio"] >= 0.45).astype(int)
df["Senior"] = (df["Age"] >= 60).astype(int)

X = df.drop("Default", axis=1)
y = df["Default"]

# 3. Encoding ---------------------------------------------------
categorical_cols = X.select_dtypes(include="object").columns
encoder = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough",
)

# Encode full dataset for CV ------------------------------------
X_encoded = encoder.fit_transform(X)

# 4. SMOTE balancing --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 5. Scale_pos_weight -------------------------------------------
pos = sum(y_train_bal)
neg = len(y_train_bal) - pos
spw = neg / pos

# 6. Model definition -------------------------------------------
xgb = XGBClassifier(
    random_state=42,
    eval_metric="auc",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1,
    scale_pos_weight=spw,
)

# 7. Train model ------------------------------------------------
xgb.fit(X_train_bal, y_train_bal)

# 8. Predict and evaluate ---------------------------------------
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred))
print("ROC‑AUC:", round(roc_auc_score(y_test, y_proba), 3))
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Paid", "Default"]).plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 9. Cross-validation ROC‑AUC ----------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = cross_val_score(xgb, X_encoded, y, cv=cv, scoring="roc_auc")
print("Cross‑val ROC‑AUC:", auc_scores.mean().round(3), "±", auc_scores.std().round(3))

# 10. Save model + encoder --------------------------------------
joblib.dump(xgb, "loan_default_model.pkl")
joblib.dump(encoder, "encoder.pkl")
print("\n✅ Saved model to loan_default_model.pkl")
print("✅ Saved encoder to encoder.pkl")


