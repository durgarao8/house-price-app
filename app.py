import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------------------------
# Custom Styling
# -------------------------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:700;
        color:#1f4e79;
    }
    .sub-title {
        font-size:18px;
        color:gray;
    }
    .stButton>button {
        background-color:#1f77b4;
        color:white;
        border-radius:8px;
        height:3em;
        width:100%;
        font-size:18px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Header Section
# -------------------------------------------------
st.markdown('<p class="main-title">🏠 California House Price Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Machine Learning powered real estate value prediction</p>', unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------
# Load & Train Model (Cached)
# -------------------------------------------------
@st.cache_resource
def train_model():
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    return model, score

model, model_score = train_model()

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("🏘 Enter House Details")

MedInc = st.sidebar.slider("Median Income", 0.0, 15.0, 3.5)
HouseAge = st.sidebar.slider("House Age", 1.0, 50.0, 20.0)
AveRooms = st.sidebar.slider("Average Rooms", 1.0, 15.0, 5.0)
AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.sidebar.slider("Population", 100.0, 5000.0, 1000.0)
AveOccup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.0)
Longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -122.0)

# -------------------------------------------------
# Main Content Layout
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Performance")
    st.metric("R² Score", f"{model_score:.2f}")

with col2:
    st.subheader("📈 About Model")
    st.write("""
    This prediction is generated using a **Random Forest Regressor**
    trained on the California Housing dataset.
    """)

st.markdown("---")

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.subheader("🔮 Predict House Price")

if st.button("Predict Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])

    prediction = model.predict(input_data)
    predicted_price = prediction[0] * 100000

    st.success("Prediction Completed Successfully!")

    st.metric(
        label="🏠 Estimated House Value",
        value=f"${predicted_price:,.2f}"
    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Developed by Durga Rao 🚀 | Machine Learning Project")
