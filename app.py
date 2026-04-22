import streamlit as st
import pandas as pd
import joblib

model = joblib.load('medical_rf_model.joblib')
model_columns = joblib.load('model_columns.joblib')

st.title("Medical Insurance Cost Predictor")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    input_encoded = pd.get_dummies(input_data, columns=['sex', 'smoker', 'region'], drop_first=True)
    
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = False
            
    input_encoded = input_encoded[model_columns]
    
    prediction = model.predict(input_encoded)[0]
    
    st.success(f"${prediction:,.2f}")