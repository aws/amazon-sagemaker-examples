""" Airline Dataset target label and feature column names  """
airline_label_column = "ArrDel15"
airline_feature_columns = [
    "Year",
    "Quarter",
    "Month",
    "DayOfWeek",
    "Flight_Number_Reporting_Airline",
    "DOT_ID_Reporting_Airline",
    "OriginCityMarketID",
    "DestCityMarketID",
    "DepTime",
    "DepDelay",
    "DepDel15",
    "ArrDel15",
    "AirTime",
    "Distance",
]
airline_dtype = "float32"

""" NYC TLC Trip Record Data target label and feature column names  """
nyctaxi_label_column = "above_average_tip"
nyctaxi_feature_columns = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "above_average_tip",
]
nyctaxi_dtype = "float32"


""" Insert your dataset here! """

BYOD_label_column = ""  # e.g., nyctaxi_label_column
BYOD_feature_columns = []  # e.g., nyctaxi_feature_columns
BYOD_dtype = None  # e.g., nyctaxi_dtype
