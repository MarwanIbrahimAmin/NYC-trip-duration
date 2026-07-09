import os
import sys
import time
import io
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

# Navigation Mode
st.sidebar.write("### Navigation")
app_mode = st.sidebar.radio("Select Prediction Mode:", ["Single Trip", "Batch File"])

if app_mode == "Single Trip":
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

else:
    # Sidebar Batch Details
    st.sidebar.write("### Batch Configuration")
    st.sidebar.markdown("""
        Upload a CSV or Excel file containing NYC taxi trip records to perform bulk predictions.
        
        The file will be processed locally using the saved champion model.
    """)
    
    st.write("### Batch File Prediction")
    st.markdown("""
        Please upload a CSV or Excel file (`.xlsx`, `.xls`) containing the required fields.
        The application will compute trip duration predictions and make the annotated file available for download.
    """)
    
    # Template layout/guide
    st.write("#### Expected File Structure")
    st.markdown("""
    The uploaded file must contain the following columns:
    - `pickup_datetime` (format: `YYYY-MM-DD HH:MM:SS`)
    - `pickup_longitude` (numeric value within NYC bounds: `[-74.25, -73.5]`)
    - `pickup_latitude` (numeric value within NYC bounds: `[40.5, 41.0]`)
    - `dropoff_longitude` (numeric value within NYC bounds: `[-74.25, -73.5]`)
    - `dropoff_latitude` (numeric value within NYC bounds: `[40.5, 41.0]`)
    
    Optional columns (sensible default values will be used if omitted):
    - `passenger_count` (defaults to `1`)
    - `vendor_id` (defaults to `1`)
    - `store_and_fwd_flag` (defaults to `"N"`)
    """)
    
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        try:
            if file_extension == ".csv":
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading the uploaded file: {str(e)}")
            input_df = None
            
        if input_df is not None:
            st.write("#### Uploaded Data Preview")
            st.dataframe(input_df.head(5), use_container_width=True)
            
            required_cols = [
                "pickup_datetime",
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude"
            ]
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in uploaded file: {', '.join(missing_cols)}")
            else:
                # Add a generate button
                predict_batch_btn = st.button("Generate Predictions")
                
                if predict_batch_btn:
                    with st.spinner("Processing predictions..."):
                        PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "Modeling/Saved_Models/preprocessor.joblib")
                        MODEL_PATH = os.path.join(PROJECT_ROOT, "Modeling/Saved_Models/best_model.joblib")
                        
                        if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
                            try:
                                local_preprocessor = joblib.load(PREPROCESSOR_PATH)
                                local_model = joblib.load(MODEL_PATH)
                                
                                # Prepare df for preprocessor
                                df_prep = input_df.copy()
                                if "trip_duration" in df_prep.columns:
                                    df_prep = df_prep.drop(columns=["trip_duration"])
                                
                                # Coerce types and handle defaults
                                df_prep["pickup_datetime"] = pd.to_datetime(df_prep["pickup_datetime"], errors="coerce")
                                df_prep["pickup_longitude"] = pd.to_numeric(df_prep["pickup_longitude"], errors="coerce")
                                df_prep["pickup_latitude"] = pd.to_numeric(df_prep["pickup_latitude"], errors="coerce")
                                df_prep["dropoff_longitude"] = pd.to_numeric(df_prep["dropoff_longitude"], errors="coerce")
                                df_prep["dropoff_latitude"] = pd.to_numeric(df_prep["dropoff_latitude"], errors="coerce")
                                
                                if "passenger_count" not in df_prep.columns:
                                    df_prep["passenger_count"] = 1
                                else:
                                    df_prep["passenger_count"] = pd.to_numeric(df_prep["passenger_count"], errors="coerce").fillna(1).astype(int)
                                    
                                if "vendor_id" not in df_prep.columns:
                                    df_prep["vendor_id"] = 1
                                else:
                                    df_prep["vendor_id"] = pd.to_numeric(df_prep["vendor_id"], errors="coerce").fillna(1).astype(int)
                                    
                                if "store_and_fwd_flag" not in df_prep.columns:
                                    df_prep["store_and_fwd_flag"] = "N"
                                else:
                                    df_prep["store_and_fwd_flag"] = df_prep["store_and_fwd_flag"].fillna("N").astype(str)
                                    
                                if "id" not in df_prep.columns:
                                    df_prep["id"] = [f"batch_{i}" for i in range(len(df_prep))]
                                    
                                # Identify rows with valid values
                                valid_mask = (
                                    df_prep["pickup_datetime"].notna() &
                                    df_prep["pickup_longitude"].notna() &
                                    df_prep["pickup_latitude"].notna() &
                                    df_prep["dropoff_longitude"].notna() &
                                    df_prep["dropoff_latitude"].notna()
                                )
                                
                                # Initialize results in output dataframe
                                output_df = input_df.copy()
                                output_df["predicted_duration_seconds"] = np.nan
                                output_df["predicted_duration_formatted"] = "Invalid Input Data"
                                output_df.loc[valid_mask, "predicted_duration_formatted"] = "Filtered (Out of NYC Bounds)"
                                
                                df_valid = df_prep[valid_mask].copy()
                                
                                if not df_valid.empty:
                                    # Transform
                                    processed_df = local_preprocessor.transform(df_valid)
                                    
                                    if not processed_df.empty:
                                        # Predict
                                        log_pred = local_model.predict(processed_df)
                                        duration_seconds = np.expm1(log_pred)
                                        
                                        # Map back using the index of the rows that survived preprocessing
                                        output_df.loc[processed_df.index, "predicted_duration_seconds"] = np.round(duration_seconds, 2)
                                        
                                        # Format output
                                        def format_row_duration(val):
                                            if pd.isna(val) or val < 0:
                                                return "Filtered (Out of NYC Bounds)"
                                            mins = int(val // 60)
                                            secs = int(val % 60)
                                            return f"{mins}m {secs}s"
                                            
                                        output_df.loc[processed_df.index, "predicted_duration_formatted"] = output_df.loc[processed_df.index, "predicted_duration_seconds"].apply(format_row_duration)
                                        
                                st.session_state["batch_results"] = output_df
                                st.session_state["batch_processed"] = True
                                st.success("Batch Prediction Completed Successfully!")
                                
                            except Exception as local_err:
                                st.error(f"Prediction failed during execution: {str(local_err)}")
                        else:
                            st.error("Prediction Service is offline and local model files were not found under Modeling/Saved_Models.")
                            
                # Show results if processed
                if st.session_state.get("batch_processed", False) and "batch_results" in st.session_state:
                    res_df = st.session_state["batch_results"]
                    total_rows = len(res_df)
                    successful_preds = res_df["predicted_duration_seconds"].notna().sum()
                    filtered_rows = total_rows - successful_preds
                    
                    st.write("### Prediction Metrics")
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.markdown(f"""
                            <div class="card">
                                <span class="metric-label">TOTAL UPLOADED</span>
                                <div class="metric-value">{total_rows}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f"""
                            <div class="card">
                                <span class="metric-label">SUCCESSFUL PREDICTIONS</span>
                                <div class="metric-value">{successful_preds}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col_m3:
                        st.markdown(f"""
                            <div class="card">
                                <span class="metric-label">FILTERED / INVALID</span>
                                <div class="metric-value">{filtered_rows}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col_m4:
                        if successful_preds > 0:
                            mean_sec = res_df["predicted_duration_seconds"].mean()
                            mean_min = int(mean_sec // 60)
                            mean_sec_rem = int(mean_sec % 60)
                            mean_formatted = f"{mean_min}m {mean_sec_rem}s"
                        else:
                            mean_formatted = "N/A"
                        st.markdown(f"""
                            <div class="card">
                                <span class="metric-label">AVERAGE DURATION</span>
                                <div class="metric-value">{mean_formatted}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    st.write("### Prediction Results Preview")
                    st.dataframe(res_df, use_container_width=True)
                    
                    # Download files
                    st.write("### Export Predictions")
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        csv_data = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_data,
                            file_name="nyc_trip_predictions.csv",
                            mime="text/csv"
                        )
                    with col_d2:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            res_df.to_excel(writer, index=False, sheet_name='Predictions')
                        excel_data = buffer.getvalue()
                        st.download_button(
                            label="Download Results as Excel",
                            data=excel_data,
                            file_name="nyc_trip_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
