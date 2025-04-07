# main.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tempfile
import requests
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

# --- Custom Transformers ---
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, input_features=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# --- Load model from external URL ---
@st.cache_resource
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download model: {response.status_code}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    model = joblib.load(tmp_path)
    os.remove(tmp_path)  # clean up
    return model

# 👇 Replace with your actual raw download link (e.g., from Hugging Face or Google Drive direct link)
MODEL_URL = "https://huggingface.co/Nikhub7-dotcom/california-housing-model/resolve/main/my_california_housing_model.pkl"

try:
    model = load_model_from_url(MODEL_URL)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- UI ---
st.title("🏠 California Housing Price Estimator")
st.markdown("**Note:** Model trained on 1990 California housing data.")

# Default values for testing
median_income = st.number_input("Median Income", value=3.5)
housing_median_age = st.number_input("Housing Median Age", value=30)
total_rooms = st.number_input("Total Rooms", value=2000)
total_bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=800)
households = st.number_input("Households", value=300)
latitude = st.number_input("Latitude", value=34.0)
longitude = st.number_input("Longitude", value=-118.0)
ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

if st.button("Estimate Price"):
    input_df = pd.DataFrame([{
        "median_income": median_income,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "latitude": latitude,
        "longitude": longitude,
        "ocean_proximity": ocean_proximity
    }])
    
    prediction = model.predict(input_df)
    st.success(f"💰 Estimated Median House Value: ${prediction[0] * 100000:.2f}")
