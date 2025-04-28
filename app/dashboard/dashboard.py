import streamlit as st
import numpy as np
import joblib
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'lightgbm_best_model.pkl')
model = joblib.load(model_path)

st.title("Predictive Maintenance Failure Prediction")

st.sidebar.header("Input Machine Metrics")

temperature = st.sidebar.slider('Temperature (°C)', 0.0, 1.0, 0.5)
pressure = st.sidebar.slider('Pressure (bar)', 0.0, 1.0, 0.5)
vibration = st.sidebar.slider('Vibration (mm/s)', 0.0, 1.0, 0.5)
rotation_speed = st.sidebar.slider('Rotation Speed (RPM)', 0.0, 1.0, 0.5)
torque = st.sidebar.slider('Torque (Nm)', 0.0, 1.0, 0.5)
oil_level = st.sidebar.slider('Oil Level (%)', 0.0, 1.0, 0.5)
load = st.sidebar.slider('Load (%)', 0.0, 1.0, 0.5)
humidity = st.sidebar.slider('Humidity (%)', 0.0, 1.0, 0.5)
voltage = st.sidebar.slider('Voltage (V)', 0.0, 1.0, 0.5)

input_features = np.array([
    temperature, pressure, vibration, rotation_speed,
    torque, oil_level, load, humidity, voltage
]).reshape(1, -1)

if st.button('Predict Failure'):
    prediction_proba = model.predict_proba(input_features)[0][1]
    st.subheader("Prediction Result")
    st.write(f"Failure Probability: **{prediction_proba*100:.2f}%**")

    if prediction_proba >= 0.3:  # Set a threshold
        st.error('⚠️ High Risk of Failure!')
    else:
        st.success('✅ Low Risk of Failure!')
