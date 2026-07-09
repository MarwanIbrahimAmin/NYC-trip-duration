import os
import sys
import time
from datetime import datetime
import json
import pandas as pd
import numpy as np
import requests
import streamlit as st
import joblib

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import pipeline to allow joblib loading if API fails
from Preprocessing.preprocessing import Preprocessing_Pipeline

# Set Page Config
st.set_page_config(
    page_title="NYC Taxi Trip Predictor",
    page_icon="Taxi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek Styling (Taxi Yellow & Dark Theme accents)
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: #ffffff;
    }
    h1 {
        color: #fbc02d;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 25px;
    }
    .stButton>button {
        background-color: #fbc02d !important;
        color: #000000 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #fdd835 !important;
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(251, 192, 45, 0.4);
    }
    .card {
        background-color: #1a1d24;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #fbc02d;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin-top: 15px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #fbc02d;
    }
    .metric-label {
        font-size: 14px;
        color: #a0aab2;
    }
    </style>
""", unsafe_allow_html=True)

# Load active model metadata
meta_path = os.path.join(PROJECT_ROOT, "Modeling/Saved_Models/best_model_meta.json")
active_model = "Machine Learning"
if os.path.exists(meta_path):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            active_model = meta.get("model_name", "Machine Learning")
    except Exception:
        pass

# App Title & Description
st.write("<h1>NYC Taxi Trip Duration Predictor</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #a0aab2; font-size: 16px; margin-bottom: 30px;'>Predict taxi ride durations in New York City with {active_model} model.</p>", unsafe_allow_html=True)

# Presets for Quick Testing in NYC
presets = {
    "Custom (Enter Below)": None,
    "Times Square to Central Park": {
        "plat": 40.7580, "plon": -73.9855,
        "dlat": 40.7851, "dlon": -73.9683
    },
    "JFK Airport to Empire State Building": {
        "plat": 40.6413, "plon": -73.7781,
        "dlat": 40.7484, "dlon": -73.9857
    },
    "Brooklyn Bridge to Wall Street": {
        "plat": 40.7061, "plon": -73.9969,
        "dlat": 40.7064, "dlon": -74.0094
    }
}

# Sidebar Controls
st.sidebar.write("### Trip Configuration")

preset_selection = st.sidebar.selectbox("Select a route preset:", list(presets.keys()))

selected_preset = presets[preset_selection]

# Set coordinates based on preset
if selected_preset:
    init_plat, init_plon = selected_preset["plat"], selected_preset["plon"]
    init_dlat, init_dlon = selected_preset["dlat"], selected_preset["dlon"]
else:
    init_plat, init_plon = 40.7580, -73.9855
    init_dlat, init_dlon = 40.7851, -73.9683

# Sidebar Fields
pickup_date = st.sidebar.date_input("Pickup Date", datetime.now().date())
pickup_time = st.sidebar.time_input("Pickup Time", datetime.now().time())

passenger_count = st.sidebar.slider("Passengers", min_value=1, max_value=9, value=1)
vendor_id = st.sidebar.selectbox("Vendor ID", options=[1, 2], index=0)
store_and_fwd_flag = st.sidebar.selectbox("Store & Forward Flag", options=["N", "Y"], index=0)

# Layout: 2 Columns (Inputs on the left, Maps and outputs on the right)
col1, col2 = st.columns([1, 1.2])

with col1:
    st.write("### Location Coordinates")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pickup_lat = st.number_input("Pickup Latitude", value=init_plat, format="%.6f", min_value=40.5, max_value=41.0)
    with col_p2:
        pickup_lon = st.number_input("Pickup Longitude", value=init_plon, format="%.6f", min_value=-74.25, max_value=-73.5)
        
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        dropoff_lat = st.number_input("Dropoff Latitude", value=init_dlat, format="%.6f", min_value=40.5, max_value=41.0)
    with col_d2:
        dropoff_lon = st.number_input("Dropoff Longitude", value=init_dlon, format="%.6f", min_value=-74.25, max_value=-73.5)

    # Check boundaries
    lat_valid = (40.5 <= pickup_lat <= 41.0) and (40.5 <= dropoff_lat <= 41.0)
    lon_valid = (-74.25 <= pickup_lon <= -73.5) and (-74.25 <= dropoff_lon <= -73.5)
    
    if not (lat_valid and lon_valid):
        st.warning("Warning: Coordinates are outside New York City bounds! (Latitude: [40.5, 41.0], Longitude: [-74.25, -73.5]). Preprocessing will filter this as an outlier.")

    # Combine Date and Time
    pickup_dt = datetime.combine(pickup_date, pickup_time).strftime("%Y-%m-%d %H:%M:%S")

    # Predict Button
    predict_btn = st.button("Predict Trip Duration")

with col2:
    st.write("### Route Preview")
    # Show pickup and dropoff points on map
    map_df = pd.DataFrame({
        "lat": [pickup_lat, dropoff_lat],
        "lon": [pickup_lon, dropoff_lon],
        "point": ["Pickup", "Dropoff"]
    })
    st.map(map_df, zoom=12)

# Calculation Trigger
if predict_btn:
    # Prepare API Request Payload
    payload = {
        "pickup_datetime": pickup_dt,
        "passenger_count": passenger_count,
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,
        "vendor_id": vendor_id,
        "store_and_fwd_flag": store_and_fwd_flag
    }
    
    with st.spinner("Calculating duration..."):
        prediction_done = False
        duration_seconds = None
        formatted_duration = None
        mode = None
        
        # 1. Try to connect to the FastAPI server first
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=3)
            if response.status_code == 200:
                res_data = response.json()
                duration_seconds = res_data["predicted_duration_seconds"]
                formatted_duration = res_data["predicted_duration_formatted"]
                mode = "FastAPI Service"
                prediction_done = True
            else:
                st.sidebar.warning(f"FastAPI Server returned code {response.status_code}. Falling back to local model loading.")
        except requests.exceptions.RequestException:
            # FastAPI server is down, fallback to local prediction
            mode = "Local Model (Fallback)"
            
        # 2. Local Fallback Execution if FastAPI failed or is offline
        if not prediction_done:
            PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "Modeling/Saved_Models/preprocessor.joblib")
            MODEL_PATH = os.path.join(PROJECT_ROOT, "Modeling/Saved_Models/best_model.joblib")
            
            if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
                try:
                    # Load models locally
                    local_preprocessor = joblib.load(PREPROCESSOR_PATH)
                    local_model = joblib.load(MODEL_PATH)
                    
                    # Convert to dataframe
                    input_df = pd.DataFrame({
                        "id": ["dummy_id"],
                        "vendor_id": [vendor_id],
                        "pickup_datetime": [pickup_dt],
                        "passenger_count": [passenger_count],
                        "pickup_longitude": [pickup_lon],
                        "pickup_latitude": [pickup_lat],
                        "dropoff_longitude": [dropoff_lon],
                        "dropoff_latitude": [dropoff_lat],
                        "store_and_fwd_flag": [store_and_fwd_flag]
                    })
                    
                    processed_df = local_preprocessor.transform(input_df)
                    
                    if processed_df.empty:
                        st.error("The coordinates are outside NYC bounds and were filtered out by the preprocessor.")
                    else:
                        log_pred = local_model.predict(processed_df)
                        duration_seconds = float(np.expm1(log_pred[0]))
                        
                        # Format output
                        mins = int(duration_seconds // 60)
                        secs = int(duration_seconds % 60)
                        formatted_duration = f"{mins}m {secs}s"
                        prediction_done = True
                except Exception as local_err:
                    st.error(f"Local fallback prediction failed: {str(local_err)}")
            else:
                st.error("Prediction Service is offline and local model files were not found under `Modeling/Saved_Models`.")
                
        # 3. Display Results
        if prediction_done:
            st.success("Prediction Completed Successfully!")
            st.markdown(f"""
                <div class="card">
                    <span class="metric-label">ESTIMATED TRIP DURATION (via {mode})</span>
                    <div class="metric-value">{formatted_duration}</div>
                    <p style="margin-top: 10px; color: #a0aab2;">
                        Total Seconds: <b>{duration_seconds:.2f} s</b><br>
                        Pickup Date/Time: <code>{pickup_dt}</code><br>
                        Total Passengers: <b>{passenger_count}</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)
