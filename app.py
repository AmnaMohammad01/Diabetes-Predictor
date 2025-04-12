import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('diabetes_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Page config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("Diabetes Risk Predictor")
st.markdown("Enter the information below to estimate diabetes risk based on lifestyle and health indicators.")

# Feature input
def user_input():
    high_bp = st.selectbox("High Blood Pressure", [0, 1])
    high_chol = st.selectbox("High Cholesterol", [0, 1])
    chol_check = st.selectbox("Cholesterol Check", [0, 1])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    smoker = st.selectbox("Smoker", [0, 1])
    phys_activity = st.selectbox("Physical Activity", [0, 1])
    fruits = st.selectbox("Consumes Fruits", [0, 1])
    veggies = st.selectbox("Consumes Vegetables", [0, 1])
    alcohol = st.selectbox("Heavy Alcohol Consumption", [0, 1])
    healthcare = st.selectbox("Has Healthcare Access", [0, 1])
    no_doc = st.selectbox("Could Not Afford Doctor", [0, 1])
    gen_health = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
    mental_health = st.slider("Days of Poor Mental Health (past 30 days)", 0, 30, 5)
    physical_health = st.slider("Days of Poor Physical Health (past 30 days)", 0, 30, 5)
    diff_walk = st.selectbox("Difficulty Walking", [0, 1])
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    age = st.slider("Age Group Code (1=18-24, 13=80+)", 1, 13, 5)
    education = st.slider("Education Level (1-6)", 1, 6, 4)
    income = st.slider("Income Level (1-8)", 1, 8, 4)

    features = [high_bp, high_chol, chol_check, bmi, smoker, phys_activity, fruits, veggies,
                alcohol, healthcare, no_doc, gen_health, mental_health, physical_health,
                diff_walk, sex, age, education, income]

    return np.array([features])

input_data = user_input()

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"High Risk of Diabetes ({probability:.2%} confidence)")
    else:
        st.success(f"Low Risk of Diabetes ({1 - probability:.2%} confidence)")