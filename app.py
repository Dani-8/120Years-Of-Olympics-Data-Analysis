import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SET UP THE PAGE AND TITLE ---
st.set_page_config(page_title="Olympic Medal Prediction", layout="wide")
st.title("Olympic Medal Prediction App")
st.write("This application helps you predict the medal chances for an athlete based on their characteristics.")
st.markdown("---")


# --- 2. EDA AND VISUALIZATIONS ---
st.header("Exploratory Data Analysis (EDA)")
st.write("This section would contain the visualizations and insights from your notebook.")

# Example placeholder plot. Replace this with your actual EDA plots.
st.subheader("Athlete Age Distribution")
fig, ax = plt.subplots(figsize=(6,4))
ages = np.random.normal(loc=25, scale=5, size=1000)
sns.histplot(ages, kde=True, ax=ax)
ax.set_xlabel("Age")
ax.set_title("Distribution of Athlete Ages (Placeholder)")
st.pyplot(fig)


# --- 3. MEDAL PREDICTION MODEL ---
st.markdown("---")
st.header("Predict Medal Chance")
st.write("Enter the athlete's details to predict their chance of winning a medal.")

# Placeholder function for the model prediction.
# You will replace this with your actual function from the notebook.
# This dummy function simulates a prediction based on the inputs.
def predict_medal_chance(age, height, weight, gender, sport, country):
    """
    Simulates the medal prediction logic.
    Replace this with your actual machine learning model code.
    """
    # Simple logic: a hypothetical elite athlete has a high chance
    if age < 30 and height > 185 and weight > 80 and sport == 'Basketball':
        return "High Chance (Predicted Gold)"
    else:
        return "Low Chance (Predicted No Medal)"


# Create input widgets for the user
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", min_value=15, max_value=60, value=25)
    height = st.slider("Height (cm)", min_value=100, max_value=250, value=200)
    weight = st.slider("Weight (kg)", min_value=30, max_value=200, value=100)

with col2:
    gender = st.selectbox("Gender", ["M", "F"])
    sport = st.selectbox("Sport", ["Basketball", "Swimming", "Athletics", "Gymnastics", "Wrestling", "Other"])
    country = st.selectbox("Country", ["USA", "China", "Russia", "Germany", "Japan", "Other"])


if st.button("Predict Medal Chance"):
    # Call the prediction function with user inputs
    prediction_result = predict_medal_chance(age, height, weight, gender, sport, country)

    st.subheader("Prediction Result")
    if "High" in prediction_result:
        st.success(f"**Prediction:** {prediction_result}")
    else:
        st.info(f"**Prediction:** {prediction_result}")
