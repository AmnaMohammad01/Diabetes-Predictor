import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('diabetes_rf_model.pkl')

# App config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("Diabetes Risk Predictor")

# Input form
with st.form("prediction_form"):
    st.subheader("Patient Details")

    bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 25.0)
    
    # Replace Age Group input with user-friendly age ranges
    age_display = st.selectbox("Age Group", [
        "18–24", "25–29", "30–34", "35–39", "40–44",
        "45–49", "50–54", "55–59", "60–64", "65–69",
        "70–74", "75–79", "80+"
    ])
    age_mapping = {
        "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4,
        "40–44": 5, "45–49": 6, "50–54": 7, "55–59": 8,
        "60–64": 9, "65–69": 10, "70–74": 11, "75–79": 12, "80+": 13
    }
    age = age_mapping[age_display]

    high_bp = st.selectbox("High Blood Pressure (1 = Yes, 0 = No)", [1, 0])
    education = st.slider("Education Level (1 = Low, 6 = High)", 1, 6, 4)
    mental_health = st.slider("Poor Mental Health Days (past 30 days)", 0, 30, 5)
    physical_health = st.slider("Poor Physical Health Days (past 30 days)", 0, 30, 5)
    diff_walk = st.selectbox("Difficulty Walking (1 = Yes, 0 = No)", [1, 0])
    sex = st.radio("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0

    submit = st.form_submit_button("Predict")

if submit:
    # Form structured input
    input_df = pd.DataFrame({
        'HighBP': [high_bp],
        'BMI': [bmi],
        'Sex': [sex],
        'Age': [age],
        'Education': [education],
        'MentHlth': [mental_health],
        'PhysHlth': [physical_health],
        'DiffWalk': [diff_walk],
        # Hidden/defaults
        'HighChol': [0],
        'CholCheck': [1],
        'Smoker': [0],
        'Stroke': [0],
        'HeartDiseaseorAttack': [0],
        'PhysActivity': [1],
        'Fruits': [1],
        'Veggies': [1],
        'HvyAlcoholConsump': [0],
        'AnyHealthcare': [1],
        'NoDocbcCost': [0],
    })

    # Feature engineering
    input_df['HealthBurden'] = input_df['MentHlth'] + input_df['PhysHlth']
    input_df['HealthBurden'] = input_df['HealthBurden'].clip(upper=30)

    # Reorder columns to match training
    column_order = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'MentHlth',
        'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
        'HealthBurden'
    ]
    for col in column_order:
        if col not in input_df.columns:
            input_df[col] = 0  # Default for unused 'Income'

    input_df = input_df[column_order]

    # Predict
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    # Result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"High Risk of Diabetes ({probability:.2%} confidence)")
    else:
        st.success(f"Low Risk of Diabetes ({1 - probability:.2%} confidence)")
