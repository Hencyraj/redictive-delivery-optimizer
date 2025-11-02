# app.py (customized for your dataset)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set up Streamlit page
st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")
st.title("🚚 Predictive Delivery Optimizer")
st.write("Predict whether a delivery will be delayed using your trained machine learning model.")

MODEL_PATH = "model/model.pkl"

# Load model
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Run train_predictor.py first to train and save the model.")
    st.stop()

model = joblib.load(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader("Upload a new delivery dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Rename columns to match training
    df.columns = [col.strip().lower() for col in df.columns]
    rename_map = {
        "order_id": "order_id",
        "carrier": "carrier",
        "promised_delivery_days": "promised_days",
        "actual_delivery_days": "actual_days",
        "delivery_status": "delivery_status",
        "quality_issue": "quality_issue",
        "customer_rating": "customer_rating",
        "delivery_cost_inr": "delivery_cost"
    }
    df.rename(columns=rename_map, inplace=True)

    # Prepare features (same as training)
    required_cols = ["carrier", "promised_days", "customer_rating", "delivery_cost", "quality_issue"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    X_new = df[required_cols]

    # Make predictions
    preds = model.predict(X_new)
    proba = model.predict_proba(X_new)[:, 1]

    df["Delay_Probability"] = np.round(proba, 2)
    df["Predicted_Status"] = np.where(preds == 1, "Delayed", "On-Time")

    st.subheader("📊 Prediction Results")
    st.dataframe(df[["order_id", "carrier", "promised_days", "actual_days", "Predicted_Status", "Delay_Probability"]].head(20))

    # Summary
    delayed_count = (df["Predicted_Status"] == "Delayed").sum()
    ontime_count = (df["Predicted_Status"] == "On-Time").sum()

    st.metric("Total Delayed", delayed_count)
    st.metric("Total On-Time", ontime_count)

    st.bar_chart(df["Predicted_Status"].value_counts())

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", csv, "predicted_delays.csv", "text/csv")

else:
    st.info("👆 Upload a CSV file to start predictions.")
