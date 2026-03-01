import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 California House Price Prediction")

# -------------------------
# User Inputs
# -------------------------

longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=20)
total_rooms = st.number_input("Total Rooms", value=1000)
total_bedrooms = st.number_input("Total Bedrooms", value=200)
population = st.number_input("Population", value=800)
households = st.number_input("Households", value=300)
median_income = st.number_input("Median Income", value=3.5)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Price"):

    # Feature engineering
    rooms_per_household = total_rooms / households if households != 0 else 0

    # Create dictionary with all features
    input_dict = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "rooms_per_household": rooms_per_household,

        # All ocean columns start as 0
        "ocean_proximity_<1H OCEAN": 0,
        "ocean_proximity_INLAND": 0,
        "ocean_proximity_ISLAND": 0,
        "ocean_proximity_NEAR BAY": 0,
        "ocean_proximity_NEAR OCEAN": 0
    }

    # Set selected ocean proximity to 1
    input_dict[f"ocean_proximity_{ocean_proximity}"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"🏡 Predicted House Price: ${prediction[0]:,.2f}")
