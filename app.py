import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# Define model paths
MODEL_DIR = "models"
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
RNN_MODEL_PATH = os.path.join(MODEL_DIR, "rnn_model.h5")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")

# Load trained models
svm_model = joblib.load(SVM_MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
rnn_model = tf.keras.models.load_model(RNN_MODEL_PATH)
lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)

# Streamlit UI
st.title("🔋 Finland Energy Consumption Forecasting")

# User inputs
year = st.number_input("Year", min_value=2020, max_value=2030, value=2024)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
day = st.number_input("Day", min_value=1, max_value=31, value=1)
hour = st.number_input("Hour", min_value=0, max_value=23, value=12)

model_choice = st.selectbox("Select Model", ["SVM", "RandomForest", "RNN", "LSTM"])

# Prediction function
def predict_energy(year, month, day, hour, model_type):
    X_input = np.array([[year, month, day, hour]])
    X_scaled = scaler.transform(X_input)

    if model_type == "SVM":
        prediction = svm_model.predict(X_scaled)
    elif model_type == "RandomForest":
        prediction = rf_model.predict(X_scaled)
    elif model_type == "RNN":
        X_reshaped = X_scaled.reshape((1, 1, 4))
        prediction = rnn_model.predict(X_reshaped).flatten()
    elif model_type == "LSTM":
        X_reshaped = X_scaled.reshape((1, 1, 4))
        prediction = lstm_model.predict(X_reshaped).flatten()
    else:
        return None

    return round(prediction[0], 2)

# Predict button
if st.button("⚡ Predict Energy Consumption"):
    prediction = predict_energy(year, month, day, hour, model_choice)
    st.success(f"🔮 Predicted Energy Consumption: {prediction} MW")
