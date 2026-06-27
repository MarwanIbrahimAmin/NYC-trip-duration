from enum import Enum

class Feature(str, Enum):
    ID = "id"
    VENDOR_ID = "vendor_id"
    PICKUP_DATETIME = "pickup_datetime"
    PASSENGER_COUNT = "passenger_count"
    PICKUP_LONGITUDE = "pickup_longitude"
    PICKUP_LATITUDE = "pickup_latitude"
    DROPOFF_LONGITUDE = "dropoff_longitude"
    DROPOFF_LATITUDE = "dropoff_latitude"
    STORE_AND_FWD_FLAG = "store_and_fwd_flag"
    TRIP_DURATION = "trip_duration"  # Target

    MANHATTAN_DIST = "manhattan_dist"
    EUCLIDEAN_DIST = "euclidean_dist"
    
    HOUR = "hour"
    DAY_OF_WEEK = "day_of_week"
    MONTH = "month"
    IS_WEEKEND = "is_weekend"
    SEASON = "season"
    
    HOUR_SIN = "hour_sin"
    HOUR_COS = "hour_cos"
    DAY_SIN = "day_sin"
    DAY_COS = "day_cos"
    MONTH_SIN = "month_sin"
    MONTH_COS = "month_cos"

    