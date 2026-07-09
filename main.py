import os
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import the Preprocessing Pipeline so joblib can deserialize it
from Preprocessing.preprocessing import Preprocessing_Pipeline

app = FastAPI(
    title="NYC Taxi Trip Duration Prediction API",
    description="An API to predict the duration of taxi trips in New York City using machine learning.",
    version="1.0.0"
)

# Load artifacts
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "Modeling/Saved_Models/preprocessor.joblib")
MODEL_PATH = os.path.join(PROJECT_ROOT, "Modeling/Saved_Models/best_model.joblib")

if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(f"Preprocessor not found at: {PREPROCESSOR_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)


class TripRequest(BaseModel):
    pickup_datetime: str = Field(..., description="Format: YYYY-MM-DD HH:MM:SS")
    passenger_count: int = Field(..., ge=1, le=9, description="Number of passengers")
    pickup_longitude: float = Field(..., description="Pickup Longitude")
    pickup_latitude: float = Field(..., description="Pickup Latitude")
    dropoff_longitude: float = Field(..., description="Dropoff Longitude")
    dropoff_latitude: float = Field(..., description="Dropoff Latitude")
    vendor_id: int = Field(default=1, description="Vendor ID")
    store_and_fwd_flag: str = Field(default="N", description="Store and forward flag (Y/N)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pickup_datetime": "2005-07-01 17:21:03",
                    "passenger_count": 1,
                    "pickup_longitude": -73.9851,
                    "pickup_latitude": 40.7589,
                    "dropoff_longitude": -73.9851,
                    "dropoff_latitude": 40.7589,
                    "vendor_id": 1,
                    "store_and_fwd_flag": "N"
                }
            ]
        }
    }


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the NYC Taxi Trip Duration Prediction API!",
        "status": "active",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.post("/predict")
def predict_trip_duration(request: TripRequest):
    # Convert input request into a DataFrame
    input_data = {
        "id": ["dummy_id"], # dummy
        "vendor_id": [request.vendor_id],
        "pickup_datetime": [request.pickup_datetime],
        "passenger_count": [request.passenger_count],
        "pickup_longitude": [request.pickup_longitude],
        "pickup_latitude": [request.pickup_latitude],
        "dropoff_longitude": [request.dropoff_longitude],
        "dropoff_latitude": [request.dropoff_latitude],
        "store_and_fwd_flag": [request.store_and_fwd_flag]
    }
    
    df = pd.DataFrame(input_data)
    
    try:
        # Preprocess the data using the pipeline
        processed_df = preprocessor.transform(df)
        
        # Check if coordinates were filtered out as outliers
        if processed_df.empty:
            raise HTTPException(
                status_code=400,
                detail="The coordinates are outside New York City bounds (Latitude: [40.5, 41.0], Longitude: [-74.25, -73.5])."
            )
            
        # Predict (model outputs log(trip_duration + 1))
        log_pred = model.predict(processed_df)
        
        # Inverse transform to get actual trip duration in seconds
        duration_seconds = float(np.expm1(log_pred[0]))
        
        # Format duration for readability
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        formatted_duration = f"{minutes}m {seconds}s"
        
        return {
            "predicted_duration_seconds": round(duration_seconds, 2),
            "predicted_duration_formatted": formatted_duration,
            "input_summary": {
                "pickup": f"({request.pickup_latitude}, {request.pickup_longitude})",
                "dropoff": f"({request.dropoff_latitude}, {request.dropoff_longitude})",
                "passengers": request.passenger_count
            }
        }
        
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)