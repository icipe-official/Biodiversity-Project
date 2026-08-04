# config.R
# Pipeline Configuration Settings & Global Variables

INPUT_DIR  <- Sys.getenv("INPUT_DIR", unset = "data/input")
OUTPUT_DIR <- Sys.getenv("OUTPUT_DIR", unset = "data/output")

# Spatial CRS Settings
AFRICA_EQUAL_AREA_CRS <- "ESRI:102022"
WGS84_CRS             <- "EPSG:4326"

# Thresholds and Smoothing Parameters
LOW_CONF_AUC_THRESHOLD <- 0.70
UNCERTAINTY_THRESHOLD  <- 0.95
AGGREGATION_FACTOR     <- 25  # 25 km neighborhood for observational support smoothing

# Ensure local directories exist
if (!dir.exists(INPUT_DIR)) dir.create(INPUT_DIR, recursive = TRUE)
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)