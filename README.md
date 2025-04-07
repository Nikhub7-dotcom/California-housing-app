# California-housing-app
It is an application that predicts the estimate median house value of a district of California based on the dataset of 1990
# 🏠 California Housing Price Prediction App

This is a machine learning web application built using **Streamlit** that predicts median house values in California districts based on various features. The model is trained on the **California Housing dataset from 1990**.

## 🚀 Features

- Predict housing prices based on user input
- Built with a trained **RandomForestRegressor** pipeline
- User-friendly interface using **Streamlit**
- Modular and extendable codebase

## 📊 About the Dataset

The data used in this project comes from the **California Housing dataset**, which is based on data from the **1990 U.S. Census**. It includes features like income, population, and geographical coordinates for each district in California.

## 🧠 Technologies Used

- Python
- Scikit-learn
- Pandas
- Numpy
- Joblib
- Streamlit

## 📁 Project Structure

. ├── main.py # Streamlit app entry point ├── housing_model_training.ipynb # Notebook used for training the model ├── .gitignore # Git ignore file ├── README.md # This file ├── chapter2env/ # Virtual environment (excluded from Git)


> ⚠️ The `.pkl` model file is not included in the repo due to GitHub's file size limit. You can retrain the model using the provided notebook.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nikhub7-dotcom/California-housing-app.git
   cd California-housing-app

python -m venv chapter2env
# Windows
chapter2env\Scripts\activate
# macOS/Linux
source chapter2env/bin/activate

pip install -r requirements.txt

streamlit run main.py

📌 Inputs Used for Prediction
Median income

Housing age

Total rooms

Total bedrooms

Population

Households

Latitude

Longitude

📝 License
This project is for educational and demonstration purposes only. Feel free to use and adapt it!


Let me know if you'd like to include deployment instructions (like using Streamlit Cloud) or a screenshot of the app!

