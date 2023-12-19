import streamlit as st
import requests

# Form Elements
with st.form("Diabetes data"):
    pregnancies = st.text_input("No of pregnancies")
    glucose = st.text_input("Glucose level")
    blood_pressure = st.text_input("Blood pressure")
    skin_thickness = st.text_input("Skin thickness")
    insulin = st.text_input("Insulin level")
    bmi = st.text_input("BMI")
    diabetes_pedigree_function = st.text_input("Diabetes pedigree function level")
    age = st.text_input("Age")

    # Submit button
    if st.form_submit_button("Send"):
        url="http://localhost:5000/predictdata"

        data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree_function,
            "Age": age
        }

        response = requests.post(url, json=data)

        st.write(response.text)

