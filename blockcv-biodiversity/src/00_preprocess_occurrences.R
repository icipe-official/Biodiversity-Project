# src/00_preprocess_occurrences.R
# Occurrence Pre-Processing, 150 km Spatial Grid Thinning, and Dataset Merging

source("config.R")

suppressPackageStartupMessages({
  library(sf)
  library(terra)
  library(dplyr)
  library(readr)
})

message("--- STAGE 0: Occurrence Pre-Processing & Grid Thinning ---")

shp_file <- file.path(INPUT_DIR, "Butterfly_Moth.shp")
file2    <- file.path(INPUT_DIR, "Butterfly_Moth_combined_thinned_5km.csv")
out_csv  <- file.path(INPUT_DIR, "Butterfly_Moth_combined_final.csv")

# 1. Spatial Grid Thinning on Shapefile (150 km grid in ESRI:102022)
if (file.exists(shp_file)) {
  message("Thinning primary shapefile using 150 km grid...")
  pts <- st_read(shp_file, quiet = TRUE) %>%
    st_transform(as.numeric(gsub("ESRI:", "", AFRICA_EQUAL_AREA_CRS)))

  grid <- st_make_grid(pts, cellsize = 150000, square = TRUE)
  grid_sf <- st_sf(id = seq_along(grid), geometry = grid)

  thinned <- st_join(pts, grid_sf) %>%
    group_by(id) %>%
    slice(1) %>%
    ungroup() %>%
    st_transform(4326)

  coords <- st_coordinates(thinned)
  df1 <- data.frame(longitude = coords[, 1], latitude = coords[, 2])
  
  cat("Original shapefile points:", nrow(pts), "\n")
  cat("150 km thinned points:", nrow(df1), "\n")
} else {
  stop("Input shapefile not found at expected path: ", shp_file)
}

# 2. Ingest Secondary 5 km Thinned Dataset
if (file.exists(file2)) {
  message("Merging with secondary 5 km dataset...")
  df2 <- read_csv(file2, show_col_types = FALSE)
  names(df2) <- tolower(names(df2))
  df2 <- df2[, c("longitude", "latitude")]
} else {
  message("Secondary dataset not found; proceeding with primary thinned set only.")
  df2 <- data.frame(longitude = numeric(0), latitude = numeric(0))
}

# 3. Combine and Deduplicate Records
names(df1) <- tolower(names(df1))
df1 <- df1[, c("longitude", "latitude")]

combined <- bind_rows(df1, df2) %>%
  filter(!is.na(longitude), !is.na(latitude)) %>%
  distinct(longitude, latitude, .keep_all = TRUE)

write_csv(combined, out_csv)

cat("Combined unique records successfully written to:", out_csv, "\n")
cat("Total Final Occurrences:", nrow(combined), "\n")