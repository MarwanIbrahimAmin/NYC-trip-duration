"""
NYC Taxi Trip Duration - Data Preprocessing Pipeline

This script formalizes the data cleaning and feature engineering workflow 
developed during the EDA phase. It transforms raw ride records into a clean, 
optimized dataset ready for Machine Learning models.

Pipeline Workflow & Implementations:
----------------------------------
1. Data Cleaning & Outlier Suppression:
   - Drops non-predictive administrative columns (e.g., 'id').
   - Filters target variable ('trip_duration') to logical bounds: [60s to 7000s].
   - Truncates spatial coordinates to strict NYC borders:
     * Latitude: [40.5, 41.0]
     * Longitude: [-74.25, -73.5]
   - Discards low-variance/redundant categories (e.g., 'store_and_fwd_flag').

2. Target Normalization:
   - Applies logarithmic scaling (np.log1p) on 'trip_duration' to correct 
     severe right-skewness and stabilize target variance.

3. Geospatial Feature Engineering:
   - Extracts vector-based displacement attributes from coordinates:
     * Manhattan Distance: Reflects NYC's urban grid block structure.
     * Euclidean Distance: Captures straight-line spatial displacement.

4. Temporal & Cyclical Transformation:
   - Parses 'pickup_datetime' into granular features: Hour, Month, Day of Week.
   - Maps monthly metrics into categorical 'Season' groups, followed by Label Encoding.
   - Generates 'is_weekend' binary flag to separate high-traffic structural shifts.
   - Encodes time boundaries (Hour, Day, Month) using Trigonometric functions 
     (Sine & Cosine transformations) to preserve chronological continuity.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PolynomialFeatures
from data.load_data import get_data_path
from data.load_data import load_csv
from data.load_data import load_train_val_test
from Enum.Feature import Feature
from Enum.path_enums import DataTier, DataFile
from Config import load_config
config = load_config()
degree = config['polynomial']['degree']
include_bias = config['polynomial']['include_bias']


class Preprocessing_Pipeline():
    def __init__(self):
        self.poly = None
        self.scaler = self._build_scaler(config.get("scaling", {}))
        self.season_encoder = LabelEncoder()
        self.scaled_columns = [
            Feature.PASSENGER_COUNT.value,
            Feature.PICKUP_LONGITUDE.value,
            Feature.PICKUP_LATITUDE.value,
            Feature.DROPOFF_LONGITUDE.value,
            Feature.DROPOFF_LATITUDE.value,
            Feature.MANHATTAN_DIST.value,
            Feature.EUCLIDEAN_DIST.value,
        ]

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        df_clean = df_clean[
            (df_clean[Feature.PICKUP_LATITUDE.value].between(40.5, 41.0)) &
            (df_clean[Feature.PICKUP_LONGITUDE.value].between(-74.25, -73.5))]
        
        if Feature.TRIP_DURATION.value in df_clean.columns:
            df_clean = df_clean[df_clean[Feature.TRIP_DURATION.value].between(60, 7000)]
            df_clean[Feature.TRIP_DURATION.value] = np.log1p(df_clean[Feature.TRIP_DURATION.value])

        return df_clean

    def _extract_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_geo = df.copy()
        
        df_geo[Feature.MANHATTAN_DIST.value] = (df_geo[Feature.DROPOFF_LATITUDE.value] - df_geo[Feature.PICKUP_LATITUDE.value]).abs() + \
                                   (df_geo[Feature.DROPOFF_LONGITUDE.value] - df_geo[Feature.PICKUP_LONGITUDE.value]).abs()
    
        df_geo[Feature.EUCLIDEAN_DIST.value] = np.sqrt(
            (df_geo[Feature.DROPOFF_LATITUDE.value] - df_geo[Feature.PICKUP_LATITUDE.value])**2 + 
            (df_geo[Feature.DROPOFF_LONGITUDE.value] - df_geo[Feature.PICKUP_LONGITUDE.value])**2
        )
        return df_geo

    def _extract_temporal_features(self, df: pd.DataFrame, train_mode: bool = True) -> pd.DataFrame:
        df_tem = df.copy()
        df_tem[Feature.PICKUP_DATETIME.value] = pd.to_datetime(df_tem[Feature.PICKUP_DATETIME.value])
        df_tem[Feature.HOUR.value] = df_tem[Feature.PICKUP_DATETIME.value].dt.hour
        df_tem[Feature.DAY_OF_WEEK.value] = df_tem[Feature.PICKUP_DATETIME.value].dt.dayofweek
        df_tem[Feature.MONTH.value] = df_tem[Feature.PICKUP_DATETIME.value].dt.month
        df_tem[Feature.IS_WEEKEND.value] = df_tem[Feature.DAY_OF_WEEK.value].isin([5, 6]).astype(int)
        df_tem[Feature.HOUR_SIN.value] = np.sin(2 * np.pi * df_tem[Feature.HOUR.value] / 24.0)
        df_tem[Feature.HOUR_COS.value] = np.cos(2 * np.pi * df_tem[Feature.HOUR.value] / 24.0)
        df_tem[Feature.DAY_SIN.value] = np.sin(2 * np.pi * df_tem[Feature.DAY_OF_WEEK.value] / 7.0)
        df_tem[Feature.DAY_COS.value] = np.cos(2 * np.pi * df_tem[Feature.DAY_OF_WEEK.value] / 7.0)
        df_tem[Feature.MONTH_SIN.value] = np.sin(2 * np.pi * df_tem[Feature.MONTH.value] / 12.0)
        df_tem[Feature.MONTH_COS.value] = np.cos(2 * np.pi * df_tem[Feature.MONTH.value] / 12.0)
        
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'autumn'
        df_tem[Feature.SEASON.value] = df_tem[Feature.MONTH.value].apply(get_season)
        
        if train_mode:
            df_tem[Feature.SEASON.value] = self.season_encoder.fit_transform(df_tem[Feature.SEASON.value])
        else:
            df_tem[Feature.SEASON.value] = self.season_encoder.transform(df_tem[Feature.SEASON.value])
            
        return df_tem

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()

        df_processed = self._clean_data(df_processed)
        df_processed = self._extract_geospatial_features(df_processed)
        df_processed = self._extract_temporal_features(df_processed , train_mode=True)
        df_processed = self.scaling(df_processed, train_mode=True)

        cols_to_drop = config.get("drop_columns", [])
        df_processed = df_processed.drop(columns=cols_to_drop , errors='ignore')

        return df_processed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()

        df_processed = self._clean_data(df_processed)
        df_processed = self._extract_geospatial_features(df_processed)
        df_processed = self._extract_temporal_features(df_processed , train_mode=False)
        df_processed = self.scaling(df_processed, train_mode=False)

        cols_to_drop = config.get("drop_columns", [])
        df_processed = df_processed.drop(columns=cols_to_drop , errors='ignore')

        return df_processed
    


    def polonomial_feature(self, x, train_mode: bool = True):
        if train_mode:
            self.poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
            x_poly = self.poly.fit_transform(x)
        else:
            if self.poly is None:
                raise ValueError("PolynomialFeatures has not been fitted yet! Run with train_mode=True first.")
            x_poly = self.poly.transform(x)
            
        return x_poly

    def _build_scaler(self, scaling_config: dict):
        scaler_type = str(scaling_config.get("type", "standard")).lower()

        if scaler_type == "standard":
            return StandardScaler()
        if scaler_type == "minmax":
            feature_range = scaling_config.get("feature_range", (0, 1))
            return MinMaxScaler(feature_range=tuple(feature_range))
        if scaler_type == "robust":
            return RobustScaler()
        if scaler_type == "maxabs":
            return MaxAbsScaler()
        if scaler_type in {"none", "null", "off"}:
            return None

        raise ValueError(
            f"Unsupported scaling type '{scaler_type}'. Supported types: standard, minmax, robust, maxabs, none."
        )
        

    def scaling(self, df: pd.DataFrame, train_mode: bool = True) -> pd.DataFrame:
        df_scaled = df.copy()
        cols = [col for col in self.scaled_columns if col in df_scaled.columns]

        if not cols or self.scaler is None:
            return df_scaled

        if train_mode:
            df_scaled[cols] = self.scaler.fit_transform(df_scaled[cols])
        else:
            df_scaled[cols] = self.scaler.transform(df_scaled[cols])

        return df_scaled

    
if __name__ == "__main__":
    train_df, val_df, test_df = load_train_val_test(tier=DataTier.SAMPLE)
    print(f"Raw Train Shape: {train_df.shape}")

    pipeline = Preprocessing_Pipeline()

    train_processed = pipeline.fit_transform(train_df)
    print(f"Processed Train Shape: {train_processed.shape}")

    val_processed = pipeline.transform(val_df)
    print(f"Processed Val Shape: {val_processed.shape}")

    print("\nFinal Columns after Preprocessing:")
    print(list(train_processed.columns))