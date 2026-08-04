############################################################
# Memory-Efficient Observational Uncertainty Mapping
# Africa-scale biodiversity modelling
############################################################

library(terra)
library(sf)

############################################################
# CONFIGURATION
############################################################

base_dir <- "/run/media/vincent/Extreme Pro/Data/variables/Africa_MS7_1km"
output_dir <- file.path(base_dir, "IBI_Outputs")

clean_occurrence_file <- file.path(
  output_dir,
  "cleaned_occurrences.gpkg"
)

ibii_raster_file <- file.path(
  "/run/media/vincent/Extreme Pro/MyBiodiversity/Biodiversity/Resubmit/SDM_Outputs/combined/Africa_MS7_1km/intactness/v2/intactness_ensemble_map_v2.tif"
)

density_file <- file.path(
  output_dir,
  "sampling_density.tif"
)

support_file <- file.path(
  output_dir,
  "observational_support.tif"
)

uncertainty_file <- file.path(
  output_dir,
  "iBII_observational_uncertainty.tif"
)

reliable_file <- file.path(
  output_dir,
  "iBII_reliable.tif"
)



############################################################
# TERRA SETTINGS
############################################################

terraOptions(
  memfrac = 0.8,
  progress = 10,
  tempdir = tempdir()
)

############################################################
# LOAD DATA
############################################################

message("Loading biodiversity raster...")

ibii <- rast(ibii_raster_file)

message("Loading occurrence data...")

occ <- st_read(
  clean_occurrence_file,
  quiet = TRUE
)

occ <- st_transform(
  occ,
  crs(ibii)
)

occ_vect <- vect(occ)

############################################################
# CREATE OCCURRENCE COUNT RASTER
############################################################

message("Rasterizing occurrence records...")

density <- rasterize(
  occ_vect,
  ibii,
  fun = "count",
  background = 0,
  filename = density_file,
  overwrite = TRUE
)

############################################################
# CREATE SMOOTH SUPPORT SURFACE
############################################################

message("Creating observational support surface...")

# For 1 km raster:
# fact=10  -> 10 km support neighbourhood
# fact=25  -> 25 km support neighbourhood
# fact=50  -> 50 km support neighbourhood

aggregation_factor <- 25

support <- aggregate(
  density,
  fact = aggregation_factor,
  fun = sum,
  na.rm = TRUE
)

support <- resample(
  support,
  ibii,
  method = "bilinear"
)

############################################################
# NORMALIZE TO 0-1
############################################################

message("Normalizing support surface...")

max_support <- global(
  support,
  "max",
  na.rm = TRUE
)[1,1]

support <- support / max_support

support <- clamp(
  support,
  lower = 0,
  upper = 1
)

names(support) <- "observational_support"

writeRaster(
  support,
  support_file,
  overwrite = TRUE,
  gdal = c(
    "COMPRESS=LZW",
    "PREDICTOR=2"
  )
)

############################################################
# CALCULATE UNCERTAINTY
############################################################

message("Calculating uncertainty...")

uncertainty <- abs(
  ibii - support
)

uncertainty <- clamp(
  uncertainty,
  lower = 0,
  upper = 1
)

names(uncertainty) <- "uncertainty"

writeRaster(
  uncertainty,
  uncertainty_file,
  overwrite = TRUE,
  gdal = c(
    "COMPRESS=LZW",
    "PREDICTOR=2"
  )
)

############################################################
# CREATE RELIABLE BIODIVERSITY MAP
############################################################

message("Masking highly uncertain regions...")

uncertainty_threshold <- 0.95

reliable_ibii <- ifel(
  uncertainty <= uncertainty_threshold,
  ibii,
  NA
)

names(reliable_ibii) <- "iBII_reliable"

writeRaster(
  reliable_ibii,
  reliable_file,
  overwrite = TRUE,
  gdal = c(
    "COMPRESS=LZW",
    "PREDICTOR=2"
  )
)

############################################################
# SUMMARY STATISTICS
############################################################

message("Calculating summary statistics...")

stats <- global(
  uncertainty,
  c("min", "mean", "max"),
  na.rm = TRUE
)

print(stats)

############################################################
# OPTIONAL UNCERTAINTY CLASSES
############################################################

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
  file.path(
    output_dir,
    "iBII_uncertainty_classes.tif"
  ),
  overwrite = TRUE,
  gdal = c(
    "COMPRESS=LZW",
    "PREDICTOR=2"
  )
)

############################################################
# FINISHED
############################################################

message("----------------------------------------")
message("Completed successfully")
message("----------------------------------------")
message("Support surface:")
message(support_file)

message("Uncertainty raster:")
message(uncertainty_file)

message("Reliable biodiversity map:")
message(reliable_file)
message("----------------------------------------")