# src/05_compute_ibii_exports.R
# Calculates Intactness Ratio, Observational Uncertainty, Masking & Visual Exporting

source("config.R")

suppressPackageStartupMessages({
  library(terra)
  library(sf)
  library(dplyr)
  library(readr)
  library(viridis)
})

terraOptions(
  memfrac = 0.8,
  progress = 10,
  tempdir = tempdir()
)

message("--- STAGE 5: Compute Intactness, Observational Uncertainty & Map Exports ---")

# ==============================================================================
# SECTION 1: LOAD SUITABILITY & CALCULATE INTACTNESS RATIO
# ==============================================================================

ibii_reliable_file <- file.path(OUTPUT_DIR, "D_ensemble_suitability.tif")
if (!file.exists(ibii_reliable_file)) {
  stop("Ensemble suitability raster not found at: ", ibii_reliable_file)
}
ibii_reliable <- rast(ibii_reliable_file)

potential_layer_file  <- file.path(INPUT_DIR, "pnv_lvl1_004_reclass.tif")
potential_output_file <- file.path(OUTPUT_DIR, "P_potential_diversity_aligned.tif")

if (file.exists(potential_layer_file)) {
  message("Aligning potential layer geometry...")
  potential_layer <- rast(potential_layer_file)

  if (crs(ibii_reliable) != crs(potential_layer)) {
    potential_aligned <- project(potential_layer, ibii_reliable, method = "near")
  } else {
    potential_aligned <- resample(potential_layer, ibii_reliable, method = "near")
  }
  
  potential_aligned <- crop(potential_aligned, ibii_reliable)
  
  writeRaster(
    potential_aligned,
    filename = potential_output_file,
    overwrite = TRUE,
    gdal = c("COMPRESS=LZW", "PREDICTOR=2")
  )
  potential_aligned <- rast(potential_output_file)
} else {
  warning("Potential layer file not found. Initializing constant baseline P = 1.0.")
  potential_aligned <- rast(ibii_reliable)
  values(potential_aligned) <- 1.0
}

message("Calculating intactness ratio...")
intactness_output_file <- file.path(OUTPUT_DIR, "intactness_ratio.tif")
ibii_stack <- c(ibii_reliable, potential_aligned)
intactness <- rast(ibii_reliable)

b <- blocks(ibii_stack)
readStart(ibii_stack)
writeStart(
  intactness,
  filename = intactness_output_file,
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "PREDICTOR=2")
)

for (i in seq_len(b$n)) {
  v <- readValues(ibii_stack, row = b$row[i], nrows = b$nrows[i], mat = TRUE)
  
  d_val <- v[, 1]
  p_val <- v[, 2]
  
  p_val[p_val <= 0] <- NA
  ratio <- d_val / p_val
  
  # Cap values strictly between 0 and 1
  ratio <- pmax(0, pmin(1, ratio))
  
  writeValues(intactness, ratio, b$row[i], b$nrows[i])
}

readStop(ibii_stack)
writeStop(intactness)

intactness <- rast(intactness_output_file)
names(intactness) <- "intactness_ratio"
message("Intactness layer successfully written to: ", intactness_output_file)

# ==============================================================================
# SECTION 2: OBSERVATIONAL UNCERTAINTY MAPPING
# ==============================================================================

message("--- Computing Observational Uncertainty Surface ---")

occ_file <- file.path(INPUT_DIR, "Butterfly_Moth_combined_final.csv")
if (!file.exists(occ_file)) {
  stop("Cleaned occurrence dataset not found at: ", occ_file)
}

message("Loading occurrence data...")
occ_df <- read_csv(occ_file, show_col_types = FALSE)
occ_sf <- st_as_sf(occ_df, coords = c("longitude", "latitude"), crs = 4326)
occ_sf <- st_transform(occ_sf, crs(intactness))
occ_vect <- vect(occ_sf)

message("Rasterizing occurrence records...")
density_file <- file.path(OUTPUT_DIR, "sampling_density.tif")
density <- rasterize(
  occ_vect,
  intactness,
  fun = "count",
  background = 0,
  filename = density_file,
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "PREDICTOR=2")
)

message("Creating observational support surface...")
support <- aggregate(
  density,
  fact = AGGREGATION_FACTOR,
  fun = sum,
  na.rm = TRUE
)

support <- resample(
  support,
  intactness,
  method = "bilinear"
)

message("Normalizing support surface...")
max_support <- global(support, "max", na.rm = TRUE)[1, 1]

if (is.na(max_support) || max_support == 0) {
  max_support <- 1
}

support <- support / max_support
support <- clamp(support, lower = 0, upper = 1)
names(support) <- "observational_support"

support_file <- file.path(OUTPUT_DIR, "observational_support.tif")
writeRaster(
  support,
  support_file,
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "PREDICTOR=2")
)

message("Calculating uncertainty layer...")
uncertainty <- abs(intactness - support)
uncertainty <- clamp(uncertainty, lower = 0, upper = 1)
names(uncertainty) <- "uncertainty"

uncertainty_file <- file.path(OUTPUT_DIR, "iBII_observational_uncertainty.tif")
writeRaster(
  uncertainty,
  uncertainty_file,
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "PREDICTOR=2")
)

message("Masking highly uncertain regions...")
reliable_ibii <- ifel(
  uncertainty <= UNCERTAINTY_THRESHOLD,
  intactness,
  NA
)
names(reliable_ibii) <- "iBII_reliable"

reliable_file <- file.path(OUTPUT_DIR, "iBII_reliable.tif")
writeRaster(
  reliable_ibii,
  reliable_file,
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "PREDICTOR=2")
)

message("Uncertainty Summary Statistics:")
stats <- global(uncertainty, c("min", "mean", "max"), na.rm = TRUE)
print(stats)

message("Classifying uncertainty surface...")
uncertainty_classes <- classify(
  uncertainty,
  matrix(
    c(
      0.0, 0.2, 1,
      0.2, 0.4, 2,
      0.4, 0.6, 3,
      0.6, 0.8, 4,
      0.8, 1.0, 5
    ),
    ncol = 3,
    byrow = TRUE
  )
)

writeRaster(
  uncertainty_classes,
  file.path(OUTPUT_DIR, "iBII_uncertainty_classes.tif"),
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "PREDICTOR=2")
)

# ==============================================================================
# SECTION 3: SPATIAL BLOCK CV MASKING & EXPORTS
# ==============================================================================

blocks_sf_file <- file.path(OUTPUT_DIR, "spatial_block_folds.gpkg")
fold_auc_file   <- file.path(OUTPUT_DIR, "fold_level_auc.csv")

if (file.exists(blocks_sf_file) && file.exists(fold_auc_file)) {
  blocks_sf <- st_read(blocks_sf_file, quiet = TRUE)
  fold_auc  <- read_csv(fold_auc_file, show_col_types = FALSE)

  blocks_auc <- blocks_sf %>%
    left_join(fold_auc, by = "fold_id") %>%
    mutate(low_confidence = ifelse(AUC < LOW_CONF_AUC_THRESHOLD, 1, 0))

  fold_auc_surface <- rasterize(vect(blocks_auc), intactness, field = "AUC", touches = TRUE)
  writeRaster(
    fold_auc_surface, 
    file.path(OUTPUT_DIR, "fold_level_AUC_surface.tif"), 
    overwrite = TRUE, 
    gdal = c("COMPRESS=LZW", "PREDICTOR=2")
  )

  low_conf_mask <- rasterize(vect(blocks_auc), intactness, field = "low_confidence", touches = TRUE)
  writeRaster(
    low_conf_mask, 
    file.path(OUTPUT_DIR, "iBII_low_confidence_mask_AUC_lt_070.tif"), 
    overwrite = TRUE, 
    gdal = c("COMPRESS=LZW", "PREDICTOR=2")
  )
}

africa_boundary_file <- file.path(INPUT_DIR, "africa_boundary.gpkg")
if (file.exists(africa_boundary_file)) {
  africa <- st_read(africa_boundary_file, quiet = TRUE)
  
  png(file.path(OUTPUT_DIR, "iBII_reliable_map.png"), width = 3000, height = 2500, res = 300)
  plot(reliable_ibii, main = "Reliable Insect Biodiversity Intactness Index (i-BII)", col = rev(terrain.colors(100)))
  plot(st_geometry(st_transform(africa, crs(reliable_ibii))), add = TRUE, border = "black", lwd = 0.5)
  dev.off()
}

message("----------------------------------------")
message("Stage 5 Execution Summary")
message("----------------------------------------")
message("Intactness layer: ", intactness_output_file)
message("Observational Support: ", support_file)
message("Observational Uncertainty: ", uncertainty_file)
message("Reliable iBII Layer: ", reliable_file)
message("----------------------------------------")