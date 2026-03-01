import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("🏠 California House Price Prediction")

@st.cache_resource
def train_model():
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    return model

model = train_model()

# User Inputs
MedInc = st.number_input("Median Income", value=3.5)
HouseAge = st.number_input("House Age", value=20.0)
AveRooms = st.number_input("Average Rooms", value=5.0)
AveBedrms = st.number_input("Average Bedrooms", value=1.0)
Population = st.number_input("Population", value=1000.0)
AveOccup = st.number_input("Average Occupancy", value=3.0)
Latitude = st.number_input("Latitude", value=37.0)
Longitude = st.number_input("Longitude", value=-122.0)

if st.button("Predict"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])

    prediction = model.predict(input_data)
    st.success(f"Predicted House Value: ${prediction[0]*100000:,.2f}")
