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


# --- 2. LOAD DATA ---
@st.cache_data
def load_data(file_path):
    """Loads and caches the Olympic dataset."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error("Error: The file 'athlete_events.csv' was not found.")
        return None

# Load the data from a CSV file. Make sure this file is in the same directory as your app.py.
# The `st.cache_data` decorator ensures the data is only loaded once.
df = load_data('athlete_events.csv')


# --- 3. EDA AND VISUALIZATIONS ---
st.header("Exploratory Data Analysis (EDA)")
st.write("This section shows key insights from the Olympic dataset.")

if df is not None:
    # Visualization 1: Athlete Age Distribution
    st.subheader("Athlete Age Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    df['Age'].dropna().hist(bins=30, ax=ax, edgecolor='black')
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Athletes")
    ax.set_title("Distribution of Athlete Ages")
    st.pyplot(fig)

    # Visualization 2: Top 10 Countries by Total Medals
    st.subheader("Top 10 Countries by Total Medals")
    country_medals = df.groupby('NOC')['Medal'].count().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=country_medals.index, y=country_medals.values, ax=ax, palette='viridis')
    ax.set_ylabel("Total Medals")
    ax.set_xlabel("Country Code")
    ax.set_title("Top 10 Countries by Total Medal Count (1896-2016)")
    st.pyplot(fig)


# --- 4. MEDAL PREDICTION MODEL ---
st.markdown("---")
st.header("Predict Medal Chance")
st.write("Enter the athlete's details to predict their chance of winning a medal.")

# A more complex placeholder function using the loaded data
def predict_medal_chance(age, height, weight, gender, sport, country):
    """
    Simulates the medal prediction logic using historical data.
    Replace this with your actual machine learning model code.
    """
    if df is None:
        return "Model not available due to data loading error."

    # Calculate a simple "medal score" based on historical data and athlete characteristics
    medal_score = 0.0

    # 1. Sport success factor
    sport_medal_count = df[(df['Sport'] == sport) & df['Medal'].notna()].shape[0]
    total_medals = df['Medal'].count()
    if total_medals > 0:
        sport_success_rate = sport_medal_count / total_medals
        medal_score += sport_success_rate * 50

    # 2. Country success factor
    country_medal_count = df[(df['NOC'] == country) & df['Medal'].notna()].shape[0]
    if total_medals > 0:
        country_success_rate = country_medal_count / total_medals
        medal_score += country_success_rate * 50

    # 3. Age factor (peak performance window)
    if 20 <= age <= 30:
        medal_score += 20
    
    # 4. Height and Weight factor (simplified)
    # This is a very simple rule, you would use a model for this
    if sport == 'Basketball' and height > 190 and weight > 90:
        medal_score += 15
    elif sport == 'Gymnastics' and height < 170 and weight < 60:
        medal_score += 15

    # Determine prediction based on score
    if medal_score > 70:
        return "High Chance (Predicted Gold)"
    elif medal_score > 40:
        return "Medium Chance (Predicted Silver/Bronze)"
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
    country = st.selectbox("Country", ["USA", "China", "RUS", "GER", "JPN", "Other"])


if st.button("Predict Medal Chance"):
    # Call the prediction function with user inputs
    prediction_result = predict_medal_chance(age, height, weight, gender, sport, country)

    st.subheader("Prediction Result")
    if "High" in prediction_result:
        st.success(f"**Prediction:** {prediction_result}")
    elif "Medium" in prediction_result:
        st.warning(f"**Prediction:** {prediction_result}")
    else:
        st.info(f"**Prediction:** {prediction_result}")
