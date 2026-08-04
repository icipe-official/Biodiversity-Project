# src/02_block_cv.R
# Spatial Block Cross-Validation Fold Allocation

source("config.R")

suppressPackageStartupMessages({
  library(sf)
  library(blockCV)
  library(dplyr)
  library(readr)
})

message("--- STAGE 2: Spatial Block Cross-Validation Setup ---")

dataset_file <- file.path(OUTPUT_DIR, "model_training_dataset.csv")

if (file.exists(dataset_file)) {
  df <- read_csv(dataset_file, show_col_types = FALSE)
  sf_data <- st_as_sf(df, coords = c("x", "y"), crs = 4326)
  
  message("Constructing spatial block cross-validation folds...")
  set.seed(42)
  spatial_folds <- cv_spatial(
    x = sf_data,
    column = "presence",
    k = 5,
    size = 250000, # 250 km block size
    selection = "random",
    iteration = 50
  )
  
  sf_data$fold_id <- spatial_folds$folds_ids
  write_csv(st_drop_geometry(sf_data), file.path(OUTPUT_DIR, "model_dataset_with_folds.csv"))
  st_write(spatial_folds$blocks, file.path(OUTPUT_DIR, "spatial_block_folds.gpkg"), delete_layer = TRUE, quiet = TRUE)
  
  message("Spatial block CV partitioning completed successfully.")
} else {
  warning("Training dataset not found. Skipping spatial block generation.")
}