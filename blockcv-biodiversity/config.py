"""Pipeline Configuration Settings & Global Variables."""

import os

INPUT_DIR = os.getenv("INPUT_DIR", "data/input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/output")

# Spatial CRS Settings
AFRICA_EQUAL_AREA_CRS = "ESRI:102022"
WGS84_CRS = "EPSG:4326"

# Thresholds and Smoothing Parameters
LOW_CONF_AUC_THRESHOLD = 0.70
UNCERTAINTY_THRESHOLD = 0.95
AGGREGATION_FACTOR = 25  # 25 km neighborhood for observational support smoothing

# Ensure local directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
