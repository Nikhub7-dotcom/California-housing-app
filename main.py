import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="California Housing Price Predictor", layout="centered")

st.title("🏡 California Housing Price Predictor")
st.write("Enter the features below to predict housing price.")

# Load the model from Hugging Face Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Nikhub7/California-housing-model",  # Your Hugging Face repo
        filename="my_california_housing_model.pkl"   # Your correct model file name
    )
    model = joblib.load(model_path)
    return model

model = load_model()

# Input form
with st.form("prediction_form"):
    MedInc = st.number_input("Median Income (10k USD)", min_value=0.0, step=0.1)
    HouseAge = st.number_input("House Age", min_value=0.0, step=1.0)
    AveRooms = st.number_input("Average Number of Rooms", min_value=0.0, step=0.1)
    AveBedrms = st.number_input("Average Number of Bedrooms", min_value=0.0, step=0.1)
    Population = st.number_input("Population", min_value=0.0, step=1.0)
    AveOccup = st.number_input("Average Occupancy", min_value=0.0, step=0.1)
    Latitude = st.number_input("Latitude", min_value=0.0, step=0.01)
    Longitude = st.number_input("Longitude", min_value=0.0, step=0.01)

    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame(
        [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]],
        columns=[
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ]
    )

    prediction = model.predict(input_data)[0]
    st.success(f"🏠 Estimated House Price: **${prediction * 100000:.2f}**")
