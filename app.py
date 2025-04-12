import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('diabetes_rf_model.pkl')

# Page configuration
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🧠 Diabetes Risk Predictor")
st.markdown("Provide the information below to estimate diabetes risk. Only the most impactful features are required.")

# Input form
with st.form("prediction_form"):
    st.subheader("User Information")

    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    age = st.slider("Age Group (1 = 18-24, 13 = 80+)", 1, 13, 5)
    high_bp = st.selectbox("High Blood Pressure", [0, 1])
    education = st.slider("Education Level (1 = Low, 6 = High)", 1, 6, 4)
    income = st.slider("Income Level (1 = Low, 8 = High)", 1, 8, 4)
    mental_health = st.slider("Days of Poor Mental Health (past 30 days)", 0, 30, 5)
    physical_health = st.slider("Days of Poor Physical Health (past 30 days)", 0, 30, 5)
    diff_walk = st.selectbox("Difficulty Walking", [0, 1])
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])

    submit = st.form_submit_button("Predict")

if submit:
    # Create input DataFrame
    input_df = pd.DataFrame({
        'HighBP': [high_bp],
        'BMI': [bmi],
        'Sex': [sex],
        'Age': [age],
        'Education': [education],
        'Income': [income],
        'MentHlth': [mental_health],
        'PhysHlth': [physical_health],
        'DiffWalk': [diff_walk],
    })

    # Fill default values for hidden features
    input_df['HighChol'] = 0
    input_df['CholCheck'] = 1
    input_df['Smoker'] = 0
    input_df['Stroke'] = 0
    input_df['HeartDiseaseorAttack'] = 0
    input_df['PhysActivity'] = 1
    input_df['Fruits'] = 1
    input_df['Veggies'] = 1
    input_df['HvyAlcoholConsump'] = 0
    input_df['AnyHealthcare'] = 1
    input_df['NoDocbcCost'] = 0

    # Derived feature
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
