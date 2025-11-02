# train_predictor.py (customized for your dataset)

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Create directories
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load dataset
file_path = "data/orders.csv"
df = pd.read_csv(file_path)

# Rename columns to simpler names
df.columns = [
    "order_id", "carrier", "promised_days", "actual_days",
    "delivery_status", "quality_issue", "customer_rating", "delivery_cost"
]

# Create target variable: 1 = delayed, 0 = on-time
df["is_delayed"] = (df["actual_days"] > df["promised_days"]).astype(int)

# Drop rows with missing important data
df = df.dropna(subset=["promised_days", "actual_days", "carrier", "delivery_cost"])

# Features and target
features = ["carrier", "promised_days", "customer_rating", "delivery_cost", "quality_issue"]
target = "is_delayed"

X = df[features]
y = df[target]

# Identify numerical and categorical columns
num_cols = ["promised_days", "customer_rating", "delivery_cost"]
cat_cols = ["carrier", "quality_issue"]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# Model pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nModel accuracy: {:.2f}%".format(acc * 100))
print("\nDetailed report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "model/model.pkl")
print("\n✅ Model saved successfully to model/model.pkl")
