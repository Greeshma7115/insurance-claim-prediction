import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("gb_model.pkl")

st.title("Insurance Claim Prediction App")

st.write("Enter customer details below:")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
region = st.selectbox("Region", [0, 1, 2, 3])

# Predict Button
if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Prediction: Likely to Claim Insurance")
    else:
        st.success(f"Prediction: Not Likely to Claim")

    st.write(f"Claim Probability: {probability:.2}")