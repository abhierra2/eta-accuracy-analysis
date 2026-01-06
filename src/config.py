"""
Configuration file for ETA analysis project.
Contains database connection settings and column name mappings.
"""

import os

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'eta_analysis',
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
}

# Connection string for SQLAlchemy
DB_CONNECTION_STRING = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# Column name mappings for different NYC taxi dataset formats
# Maps common variations to our standard column names
COLUMN_MAPPING = {
    # Standard column names we expect
    'pickup_datetime': [
        'pickup_datetime', 'tpep_pickup_datetime', 'lpep_pickup_datetime',
        'pickup_time', 'trip_pickup_datetime'
    ],
    'dropoff_datetime': [
        'dropoff_datetime', 'tpep_dropoff_datetime', 'lpep_dropoff_datetime',
        'dropoff_time', 'trip_dropoff_datetime'
    ],
    'pickup_latitude': [
        'pickup_latitude', 'start_lat', 'pickup_lat'
    ],
    'pickup_longitude': [
        'pickup_longitude', 'start_lon', 'pickup_lon'
    ],
    'dropoff_latitude': [
        'dropoff_latitude', 'end_lat', 'dropoff_lat'
    ],
    'dropoff_longitude': [
        'dropoff_longitude', 'end_lon', 'dropoff_lon'
    ],
    'trip_distance': [
        'trip_distance', 'distance', 'trip_distance_miles', 'distance_miles'
    ]
}

# Data quality thresholds
DATA_QUALITY = {
    'min_duration_sec': 30,  # Minimum trip duration (30 seconds)
    'max_duration_sec': 7200,  # Maximum trip duration (2 hours)
    'min_distance_km': 0.1,  # Minimum trip distance (100 meters)
    'max_distance_km': 100,  # Maximum trip distance (100 km)
    'valid_lat_range': (-90, 90),
    'valid_lon_range': (-180, 180),
    'nyc_lat_range': (40.4, 40.9),  # Approximate NYC bounds
    'nyc_lon_range': (-74.3, -73.7)
}

# Distance buckets for segmentation (in kilometers)
DISTANCE_BUCKETS = [
    (0, 1.609),      # < 1 mile
    (1.609, 4.828),  # 1-3 miles
    (4.828, 8.047),  # 3-5 miles
    (8.047, 16.093), # 5-10 miles
    (16.093, float('inf'))  # 10+ miles
]

DISTANCE_BUCKET_LABELS = [
    '<1mi',
    '1-3mi',
    '3-5mi',
    '5-10mi',
    '10+mi'
]

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
METRICS_DIR = os.path.join(OUTPUTS_DIR, 'metrics')

