import os
import sys
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Preprocessing.preprocessing import Preprocessing_Pipeline
from Enum.Feature import Feature

@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        Feature.ID.value: ["id1", "id2", "id3"],
        Feature.PICKUP_DATETIME.value: ["2026-07-08 17:00:00", "2026-07-08 18:30:00", "2026-07-08 23:45:00"],
        Feature.PASSENGER_COUNT.value: [1, 2, 4],
        Feature.PICKUP_LATITUDE.value: [40.7128, 40.7589, 35.0000],  # Third point is completely outside NYC
        Feature.PICKUP_LONGITUDE.value: [-74.0060, -73.9851, -120.0000], # Third point is outside NYC bounding box
        Feature.DROPOFF_LATITUDE.value: [40.7128, 40.7589, 35.0500],
        Feature.DROPOFF_LONGITUDE.value: [-73.0060, -73.8851, -120.0500],
        Feature.STORE_AND_FWD_FLAG.value: ["N", "N", "Y"],
        Feature.TRIP_DURATION.value: [500, 1200, 8000] # Trip outside temporal or spatial bounds
    })

def test_clean_data_outlier_filtering(sample_raw_data):
    """Verifies that the _clean_data function filters out trips outside NYC geographic boundaries."""
    pipeline = Preprocessing_Pipeline()
    cleaned = pipeline._clean_data(sample_raw_data, train_mode=True)
    
    # The 3rd point is located at latitude 35 and longitude -120, well outside NYC boundaries.
    # Therefore, the function should drop this row, leaving exactly 2 rows.
    assert len(cleaned) == 2, "Geographic outlier filtering failed to drop out-of-bounds coordinates."

def test_geospatial_features_calculation():
    """Verifies the mathematical correctness of Manhattan and Euclidean distance calculations."""
    pipeline = Preprocessing_Pipeline()
    
    # Simple test trip designed for easy manual verification
    df = pd.DataFrame({
        Feature.PICKUP_LATITUDE.value: [40.0],
        Feature.PICKUP_LONGITUDE.value: [-74.0],
        Feature.DROPOFF_LATITUDE.value: [40.3],
        Feature.DROPOFF_LONGITUDE.value: [-74.4]
    })
    
    geo_df = pipeline._extract_geospatial_features(df)
    
    # Manual distance calculations:
    # Manhattan = |40.3 - 40.0| + |-74.4 - (-74.0)| = 0.3 + 0.4 = 0.7
    # Euclidean = sqrt( (0.3)^2 + (-0.4)^2 ) = sqrt(0.09 + 0.16) = sqrt(0.25) = 0.5
    assert geo_df[Feature.MANHATTAN_DIST.value].iloc[0] == pytest.approx(0.7)
    assert geo_df[Feature.EUCLIDEAN_DIST.value].iloc[0] == pytest.approx(0.5)

def test_temporal_features_extraction():
    """Verifies temporal feature extraction, weekend detection, and cyclical components."""
    pipeline = Preprocessing_Pipeline()
    
    # 2026-07-08 is a Wednesday (Days 0-6, Wednesday = 2), which is not a weekend.
    df = pd.DataFrame({
        Feature.PICKUP_DATETIME.value: ["2026-07-08 12:00:00"]
    })
    
    temp_df = pipeline._extract_temporal_features(df, train_mode=True)
    
    assert temp_df[Feature.HOUR.value].iloc[0] == 12
    assert temp_df[Feature.MONTH.value].iloc[0] == 7
    assert temp_df[Feature.DAY_OF_WEEK.value].iloc[0] == 2
    assert temp_df[Feature.IS_WEEKEND.value].iloc[0] == 0  # Wednesday is not a weekend