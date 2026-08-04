# R/utils.R
# Utility functions for spatial raster operations and data handling

suppressPackageStartupMessages({
  library(terra)
  library(sf)
})

#' Apply spatial vector mask to raster safely and write to disk
#'
#' @param input_raster SpatRaster object
#' @param mask_vector sf or SpatVector object
#' @param output_path File path to write the output GTiff
#' @return SpatRaster masked object
apply_spatial_mask <- function(input_raster, mask_vector, output_path) {
  masked <- mask(input_raster, vect(mask_vector))
  writeRaster(
    masked,
    filename = output_path,
    overwrite = TRUE,
    gdal = c("COMPRESS=LZW", "PREDICTOR=2")
  )
  return(masked)
}

#' Standardize coordinate column names in a data.frame
#'
#' @param df Input data frame
#' @return Data frame with lower-case latitude/longitude names
standardize_coords <- function(df) {
  names(df) <- tolower(names(df))
  if ("lon" %in% names(df)) names(df)[names(df) == "lon"] <- "longitude"
  if ("lat" %in% names(df)) names(df)[names(df) == "lat"] <- "latitude"
  return(df)
}