import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('diabetes_rf_model.pkl')

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("🧠 Diabetes Risk Predictor")
st.markdown("Enter your details below to estimate the likelihood of developing diabetes.")

# Input form
with st.form("prediction_form"):
    st.subheader("Lifestyle & Health Information")

    high_bp = st.selectbox("High Blood Pressure", [0, 1])
    high_chol = st.selectbox("High Cholesterol", [0, 1])
    chol_check = st.selectbox("Cholesterol Check in Last 5 Years", [0, 1])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    smoker = st.selectbox("Smoker", [0, 1])
    stroke = st.selectbox("History of Stroke", [0, 1])
    heart_disease = st.selectbox("Heart Disease or Attack", [0, 1])
    phys_activity = st.selectbox("Physically Active", [0, 1])
    fruits = st.selectbox("Consumes Fruits", [0, 1])
    veggies = st.selectbox("Consumes Vegetables", [0, 1])
    alcohol = st.selectbox("Heavy Alcohol Consumption", [0, 1])
    healthcare = st.selectbox("Has Healthcare Access", [0, 1])
    no_doc = st.selectbox("Could Not Afford Doctor", [0, 1])
    mental_health = st.slider("Days of Poor Mental Health (last 30 days)", 0, 30, 5)
    physical_health = st.slider("Days of Poor Physical Health (last 30 days)", 0, 30, 5)
    diff_walk = st.selectbox("Difficulty Walking", [0, 1])
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    age = st.slider("Age Group Code (1 = 18-24, 13 = 80+)", 1, 13, 5)
    education = st.slider("Education Level (1 = Low, 6 = High)", 1, 6, 4)
    income = st.slider("Income Level (1 = Low, 8 = High)", 1, 8, 4)

    submit = st.form_submit_button("Predict")

if submit:
    # Create DataFrame
    input_df = pd.DataFrame({
        'HighBP': [high_bp],
        'HighChol': [high_chol],
        'CholCheck': [chol_check],
        'BMI': [bmi],
        'Smoker': [smoker],
        'Stroke': [stroke],
        'HeartDiseaseorAttack': [heart_disease],
        'PhysActivity': [phys_activity],
        'Fruits': [fruits],
        'Veggies': [veggies],
        'HvyAlcoholConsump': [alcohol],
        'AnyHealthcare': [healthcare],
        'NoDocbcCost': [no_doc],
        'MentHlth': [mental_health],
        'PhysHlth': [physical_health],
        'DiffWalk': [diff_walk],
        'Sex': [sex],
        'Age': [age],
        'Education': [education],
        'Income': [income]
    })

    # Add engineered feature
    input_df['HealthBurden'] = input_df['MentHlth'] + input_df['PhysHlth']
    input_df['HealthBurden'] = input_df['HealthBurden'].clip(upper=30)

    # Predict
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    # Output
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"High Risk of Diabetes ({probability:.2%} confidence)")
    else:
        st.success(f"Low Risk of Diabetes ({1 - probability:.2%} confidence)")
