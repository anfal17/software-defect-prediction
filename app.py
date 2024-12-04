import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('bug_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define Streamlit user interface
st.title("Software Bug Prediction")

st.subheader("Model Input:")
LOC = st.number_input("Lines of Code (LOC)", min_value=0)
complexity = st.number_input("Cyclomatic Complexity", min_value=0)
methods = st.number_input("Methods Count", min_value=0)

# Prepare the input for prediction
input_data = [[LOC, complexity, methods]]
input_scaled = scaler.transform(input_data)

# Make prediction using the loaded model
prediction = model.predict(input_scaled)

# Display prediction
if prediction == 1:
    st.write("This code is likely to be buggy.")
else:
    st.write("This code is likely to be bug-free.")
