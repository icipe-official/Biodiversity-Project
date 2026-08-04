library(terra)

# 1. Load layers
ibii_reliable_path   <- "/run/media/vincent/Extreme Pro/MyBiodiversity/Biodiversity/Resubmit/SDM_Outputs/combined/Africa_MS7_1km/intactness/v2/intactness_ensemble_map_v2.tif"
potential_layer_path <- "/run/media/vincent/Extreme Pro/MyBiodiversity/Biodiversity/Potential/pnv_lvl1_004_reclass.tif"

ibii_reliable   <- rast(ibii_reliable_path)
potential_layer <- rast(potential_layer_path)

# 2. Align potential_layer to match ibii_reliable geometry
message("Aligning potential layer to target geometry...")

# Check if CRS matches first
if (crs(ibii_reliable) != crs(potential_layer)) {
  # Reproject and resample if CRS differs
  potential_aligned <- project(potential_layer, ibii_reliable, method = "near")
} else {
  
  # Resample directly if CRS already matches but extents/resolutions differ
  potential_aligned <- resample(potential_layer, ibii_reliable, method = "near")
}

# Crop to exact extent boundary
potential_aligned <- crop(potential_aligned, ibii_reliable)

# 3. Calculate Intactness Ratio
# Note: Handle division by zero or background zeros in potential layer
potential_aligned[potential_aligned == 0] <- NA

intactness <- ibii_reliable / potential_aligned

# --- NEW STEP: Cap values above 1 to 1 ---
intactness[intactness > 1] <- 1

names(intactness) <- "intactness_ratio"

# 4. Save output
output_path <- "/run/media/vincent/Extreme Pro/Data/variables/Africa_MS7_1km/IBI_Outputs/intactness_ratio.tif"
writeRaster(
  intactness, 
  filename = output_path, 
  overwrite = TRUE, 
  gdal = c("COMPRESS=LZW", "PREDICTOR=2")
)

message("Intactness layer successfully written to: ", output_path)