
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib



import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.markdown("""
    <style>
        .card {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .sidebar .stSelectbox label, .sidebar .stNumberInput label {
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


st.title("📺 YouTube Ad Revenue Predictor")

# Load trained pipeline
try:
    

    pipeline = joblib.load(r"C:\Users\Manisha\OneDrive\Desktop\Projects\Linear Regression.pkl")  # Make sure this file exists


except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


st.markdown("<p style='text-align:center; color:rgba(255,255,255,0.8)'>Predict your estimated ad revenue based on video performance.</p>", unsafe_allow_html=True)
st.write("---")

# Sidebar Inputs
st.sidebar.header("🎯 Enter Inputs")

views = st.sidebar.number_input("Views", min_value=0, step=1000, value=10000)
likes = st.sidebar.number_input("Likes", min_value=0, step=100, value=500)
comments = st.sidebar.number_input("Comments", min_value=0, step=10, value=50)
watch_time_minutes = st.sidebar.number_input("Watch Time (minutes)", min_value=0, step=1, value=500)
video_length_minutes = st.sidebar.number_input("Video Length (minutes)", min_value=0, step=1, value=10)
subscribers = st.sidebar.number_input("Subscribers", min_value=0, step=100, value=10000)

category = st.sidebar.selectbox("Category", ["Entertainment", "Education", "Gaming", "Music", "Technology"])
device = st.sidebar.selectbox("Device", ["Mobile", "Desktop", "Tablet"])
country = st.sidebar.selectbox("Country", ["US", "India", "UK", "Canada", "Other"])

st.subheader("📋 Input Summary")

# -------------------------
# Prediction section
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

if st.button("🔮 Predict Revenue"):
    # Prepare input
    input_dict = {
        "views": views,
        "likes": likes,
        "comments": comments,
        "watch_time_minutes": watch_time_minutes,
        "video_length_minutes": video_length_minutes,
        "subscribers": subscribers,
        "category": category,
        "device": device,
        "country": country,
        "engagement_rate": 0.05,  # default or estimated
        "subscriber_value": 0.10, 
    }
    input_df = pd.DataFrame([input_dict])

    
    try:
        model_features = pipeline.feature_names_in_
    except Exception:
        st.error("The model does not have `feature_names_in_`. Retrain with sklearn for compatibility.")
        st.stop()

    input_encoded = input_df.reindex(columns=model_features, fill_value=0)

    # ✅ Ensure numeric columns are properly typed
    numeric_cols = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers']
    for col in numeric_cols:
        if col in input_encoded.columns:
            input_encoded[col] = pd.to_numeric(input_encoded[col], errors='coerce')

    if input_encoded[numeric_cols].isnull().any().any():
        st.error("Some numeric inputs are invalid or missing. Please check your entries.")
        st.stop()

    try:
        prediction = pipeline.predict(input_encoded)[0]
        st.success(f"💰 Estimated Ad Revenue: **${prediction:.2f} USD**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
