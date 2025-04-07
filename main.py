import streamlit as st
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# Download the model from Hugging Face if not already cached
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="Nikhub7/California-housing-model", filename="california_model.pkl")
    return joblib.load(model_path)

model = load_model()

# Streamlit App UI
st.title("California Housing Price Prediction")

st.write("Enter the details to predict the median house value:")

# Inputs
median_income = st.number_input("Median Income", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
house_age = st.number_input("House Age", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
avg_rooms = st.number_input("Average Rooms", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
avg_bedrooms = st.number_input("Average Bedrooms", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
population = st.number_input("Population", min_value=0, max_value=10000, value=1000, step=10)
households = st.number_input("Households", min_value=0, max_value=5000, value=500, step=10)
latitude = st.number_input("Latitude", min_value=30.0, max_value=50.0, value=34.0, step=0.1)
longitude = st.number_input("Longitude", min_value=-125.0, max_value=-110.0, value=-118.0, step=0.1)

# Prediction
if st.button("Predict"):
    input_data = np.array([[median_income, house_age, avg_rooms, avg_bedrooms,
                            population, households, latitude, longitude]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Median House Value: ${prediction * 100000:.2f}")
