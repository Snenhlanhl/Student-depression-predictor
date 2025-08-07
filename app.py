import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
pipeline = joblib.load("depression_model_pipeline.pkl")  # Match the filename you used in Colab

# Title
st.title("Student Depression Predictor")

st.markdown("Enter the details below to predict the risk of depression:")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100)
study_hours = st.slider("Study Hours per Day", 0, 15, 5)
sleep_hours = st.slider("Sleep Hours per Night", 0, 15, 7)
internet_use = st.selectbox("Internet Access", ["Yes", "No"])

# Create DataFrame from user input
input_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Study Hours": [study_hours],
    "Sleep Hours": [sleep_hours],
    "Internet Access": [internet_use]
})

# Predict
if st.button("Predict"):
    try:
        prediction = pipeline.predict(input_data)
        probabilities = pipeline.predict_proba(input_data)

        st.subheader("Prediction Result")
        st.write("**Depression Risk:**", "Yes" if prediction[0] == 1 else "No")
        st.write(f"**Probability of No Depression:** {probabilities[0][0]:.2f}")
        st.write(f"**Probability of Depression:** {probabilities[0][1]:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")








