# src/01_data_prep.R
# Prepares environmental predictors, background points, and extracts modeling tables

source("config.R")
source("R/utils.R")

suppressPackageStartupMessages({
  library(terra)
  library(sf)
  library(dplyr)
  library(readr)
})

# Ensure occurrence file exists before continuing
occ_file <- file.path(INPUT_DIR, "Butterfly_Moth_combined_final.csv")
if (!file.exists(occ_file)) {
  message("Combined occurrence file not found. Running pre-processing stage first...")
  source("src/00_preprocess_occurrences.R", local = new.env())
}

message("--- STAGE 1: Data Preparation ---")

covariate_file <- file.path(INPUT_DIR, "environmental_covariates.tif")
occ_df <- read_csv(occ_file, show_col_types = FALSE)

if (file.exists(covariate_file)) {
  message("Extracting environmental values at presence locations...")
  covs <- rast(covariate_file)
  occ_sf <- st_as_sf(occ_df, coords = c("longitude", "latitude"), crs = 4326)
  
  if (crs(covs) != crs(occ_sf)) {
    occ_sf <- st_transform(occ_sf, crs(covs))
  }
  
  presence_vals <- extract(covs, vect(occ_sf), ID = FALSE)
  presence_data <- cbind(st_coordinates(occ_sf), presence_vals, presence = 1)
  
  message("Sampling pseudo-absence background points...")
  set.seed(42)
  bg_pts <- spatSample(covs, size = nrow(occ_df) * 2, method = "random", values = TRUE, xy = TRUE, na.rm = TRUE)
  bg_data <- data.frame(bg_pts)
  bg_data$presence <- 0
  
  full_dataset <- bind_rows(presence_data, bg_data)
  write_csv(full_dataset, file.path(OUTPUT_DIR, "model_training_dataset.csv"))
  message("Model training dataset generated successfully.")
} else {
  warning("Environmental covariates stack not found at 'data/input/environmental_covariates.tif'. Skipping extraction step.")
}