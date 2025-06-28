import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import pathlib

# Load data
DATA_PATH = pathlib.Path("data") / "Loan_default.csv"
df = pd.read_csv(DATA_PATH)

# Drop LoanID
df.drop(columns=["LoanID"], inplace=True)

# Separate features and target
X = df.drop("Default", axis=1)
y = df["Default"]

# Identify column types
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# Define preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# Create ML pipeline
pipeline = Pipeline(steps=[
    ("preprocessing", preprocess),
    ("model", GradientBoostingClassifier(random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("✅ AUC-ROC Score:", roc_auc_score(y_test, y_proba))

# Save model
joblib.dump(pipeline, "loan_default_model.pkl")
print("\n✅ Model saved as loan_default_model.pkl")

