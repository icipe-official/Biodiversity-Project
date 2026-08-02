library(terra)

# ============================================================
# INPUT FILES
# ============================================================

raster_file <- "~/Development/geospatial-analysis/Biodiversity-Project/block-validation/blockcv-biodiversity/data/ensemble_map_v2.tif"

occurrence_file <- "~/Development/geospatial-analysis/Biodiversity-Project/block-validation/blockcv-biodiversity/data/Butterfly_Moth.shp"

setwd("~/Development/geospatial-analysis/Biodiversity-Project/block-validation/blockcv-biodiversity/data")

# ============================================================
# PARAMETERS
# ============================================================

occurrence_threshold <- 0.86
hotspot_threshold <- 0.86
n_candidate_points <- 3000
thin_distance <- 5000   # metres
random_seed <- 123

# ============================================================
# LOAD DATA
# ============================================================

reference_layer <- terra::rast(raster_file)
occ_vect <- terra::vect(occurrence_file)

cat("\nOriginal occurrences:", nrow(occ_vect), "\n")

# ============================================================
# MATCH CRS
# ============================================================

if (!terra::same.crs(reference_layer, occ_vect)) {
  occ_vect <- terra::project(
    occ_vect,
    terra::crs(reference_layer)
  )
}

# ============================================================
# EXTRACT RASTER VALUES
# ============================================================

vals <- terra::extract(
  reference_layer,
  occ_vect
)

occ_vect$raster_value <- vals[[2]]

# ============================================================
# FILTER OBSERVED OCCURRENCES
# ============================================================

occ_filtered <- occ_vect[
  !is.na(occ_vect$raster_value) &
    occ_vect$raster_value >= occurrence_threshold,
]

occ_filtered$type <- "observed"

cat(
  "Occurrences after filtering:",
  nrow(occ_filtered),
  "\n"
)

# ============================================================
# SAVE FILTERED OCCURRENCES
# ============================================================

terra::writeVector(
  occ_filtered,
  "Butterfly_Moth_filtered_086.shp",
  overwrite = TRUE
)

write.csv(
  terra::as.data.frame(
    occ_filtered,
    geom = "XY"
  ),
  "Butterfly_Moth_filtered_086.csv",
  row.names = FALSE
)

# ============================================================
# GENERATE HOTSPOT CANDIDATE POINTS
# ============================================================

set.seed(random_seed)

high_cells <- reference_layer >= hotspot_threshold
high_cells[high_cells == 0] <- NA

new_points <- terra::spatSample(
  high_cells,
  size = n_candidate_points,
  method = "random",
  as.points = TRUE,
  na.rm = TRUE
)

new_points$type <- "synthetic"

cat(
  "Synthetic hotspot points:",
  nrow(new_points),
  "\n"
)

# ============================================================
# SAVE HOTSPOT POINTS
# ============================================================

terra::writeVector(
  new_points,
  "candidate_survey_points.shp",
  overwrite = TRUE
)

write.csv(
  terra::as.data.frame(
    new_points,
    geom = "XY"
  ),
  "candidate_survey_points.csv",
  row.names = FALSE
)

# ============================================================
# COMBINE DATASETS
# ============================================================

combined_points <- rbind(
  occ_filtered,
  new_points
)

cat(
  "Combined points:",
  nrow(combined_points),
  "\n"
)

# ============================================================
# PROJECT TO METRIC CRS
# Africa Albers Equal Area
# ============================================================

combined_metric <- terra::project(
  combined_points,
  "+proj=aea +lat_1=-18 +lat_2=21 +lat_0=0 +lon_0=20 +datum=WGS84 +units=m +no_defs"
)

# ============================================================
# CREATE 5 km GRID
# ============================================================

grid <- terra::rast(
  ext(combined_metric),
  resolution = thin_distance,
  crs = crs(combined_metric)
)

# ============================================================
# ASSIGN CELL IDs
# ============================================================

cell_id <- terra::cellFromXY(
  grid,
  terra::crds(combined_metric)
)

combined_metric$cell_id <- cell_id

# ============================================================
# RANDOMLY KEEP ONE POINT PER CELL
# ============================================================

set.seed(random_seed)

df <- terra::as.data.frame(
  combined_metric,
  geom = "XY"
)

df$rand <- runif(nrow(df))

df <- df[
  order(df$cell_id, df$rand),
]

df <- df[
  !duplicated(df$cell_id),
]

# ============================================================
# REBUILD SPATVECTOR
# ============================================================

combined_thinned <- terra::vect(
  df,
  geom = c("x", "y"),
  crs = crs(combined_metric)
)

# ============================================================
# RETURN TO ORIGINAL CRS
# ============================================================

combined_thinned <- terra::project(
  combined_thinned,
  terra::crs(reference_layer)
)

# ============================================================
# SUMMARY
# ============================================================

cat("\n===========================\n")
cat("FINAL SUMMARY\n")
cat("===========================\n")
cat("Original occurrences :", nrow(occ_vect), "\n")
cat("Filtered occurrences :", nrow(occ_filtered), "\n")
cat("Synthetic points     :", nrow(new_points), "\n")
cat("Combined points      :", nrow(combined_points), "\n")
cat("After 5 km thinning  :", nrow(combined_thinned), "\n")
cat("===========================\n")

# ============================================================
# SAVE FINAL THINNED DATASET
# ============================================================

terra::writeVector(
  combined_thinned,
  "Butterfly_Moth_combined_thinned_5km.shp",
  overwrite = TRUE
)

write.csv(
  terra::as.data.frame(
    combined_thinned,
    geom = "XY"
  ),
  "Butterfly_Moth_combined_thinned_5km.csv",
  row.names = FALSE
)

# ============================================================
# FINAL MAP
# ============================================================

png(
  "Butterfly_Moth_combined_thinned_5km.png",
  width = 5000,
  height = 3500,
  res = 300
)

plot(
  reference_layer,
  main = paste0(
    "Final Combined Dataset\n",
    "5 km Spatial Thinning\n",
    "Points = ",
    format(nrow(combined_thinned), big.mark = ",")
  )
)

points(
  combined_thinned,
  pch = 16,
  col = "darkgreen",
  cex = 0.45
)

dev.off()

plot(
  reference_layer,
  main = paste0(
    "Final Combined Dataset\n",
    "5 km Spatial Thinning\n",
    "Points = ",
    format(nrow(combined_thinned), big.mark = ",")
  )
)

points(
  combined_thinned,
  pch = 16,
  col = "darkgreen",
  cex = 0.45
)

cat("\nOutputs created:\n")
cat("Butterfly_Moth_filtered_086.shp\n")
cat("candidate_survey_points.shp\n")
cat("Butterfly_Moth_combined_thinned_5km.shp\n")
cat("Butterfly_Moth_combined_thinned_5km.csv\n")
cat("Butterfly_Moth_combined_thinned_5km.png\n")