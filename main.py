# main.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

# -------------------- Custom Code --------------------

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(transformer, feature_names_in):
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
        return [f"cluster_sim_{i}" for i in range(self.n_clusters)]

# -------------------- Streamlit UI --------------------

st.title("California Housing Price Estimator")

try:
    model = joblib.load("my_california_housing_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input fields
median_income = st.number_input("Median Income", min_value=0.0, value=3.5)
total_rooms = st.number_input("Total Rooms", min_value=1.0, value=1000.0)
total_bedrooms = st.number_input("Total Bedrooms", min_value=1.0, value=200.0)
population = st.number_input("Population", min_value=1.0, value=500.0)
households = st.number_input("Households", min_value=1.0, value=300.0)
housing_median_age = st.number_input("Housing Median Age", min_value=1.0, value=25.0)
latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=36.5)
longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-120.0)
ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

# Prediction
if st.button("Estimate Price"):
    input_data = pd.DataFrame([{
        "median_income": median_income,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "housing_median_age": housing_median_age,
        "latitude": latitude,
        "longitude": longitude,
        "ocean_proximity": ocean_proximity
    }])

    try:
        prediction = model.predict(input_data)
        st.success(f"Estimated Median House Value: ${prediction[0] * 100000:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
