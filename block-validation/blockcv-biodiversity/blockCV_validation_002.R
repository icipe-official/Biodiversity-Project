############################################################
# Insect-Based Biodiversity Intactness Index (i-BII)
# Fully Chunked & Memory-Optimized Workflow
# Ensemble: Random Forest, XGBoost, MaxEnt (maxnet)
# Spatial Validation: 5 Latitudinal Folds
############################################################

# Set Working Directory safely
suppressWarnings(try(setwd("/home/vincent/Development/geospatial-analysis/Biodiversity-Project/block-validation/blockcv-biodiversity"), silent = TRUE))

############################################################
# 0. Install and load packages
############################################################

packages <- c(
  "terra",
  "sf",
  "dplyr",
  "readr",
  "ggplot2",
  "tidyr",
  "pROC",
  "caret",
  "randomForest",
  "xgboost",
  "maxnet",
  "viridis",
  "rnaturalearthdata"
)

new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) {
  install.packages(new_packages, dependencies = TRUE)
}

invisible(lapply(packages, library, character.only = TRUE))

set.seed(123)

############################################################
# 1. User settings
############################################################

# Optimize terra memory usage and force LZW compression globally
terraOptions(
  memfrac = 0.6,
  gdal = c("COMPRESS=LZW", "PREDICTOR=2")
)

# Folder structure
base_dir <- "/run/media/vincent/Extreme Pro/Data/variables/Africa_MS7_1km"

input_dir <- file.path(base_dir, "clipped")
output_dir <- file.path(base_dir, "IBI_Outputs")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Input files
occurrence_file <- file.path("~/Development/geospatial-analysis/Biodiversity-Project/block-validation/blockcv-biodiversity/data/Butterfly_Moth_combined_final.csv")
africa_boundary_file <- file.path(input_dir, "africa_boundary.gpkg")

# Predictor rasters
predictor_files <- c(
  canopy_height   = file.path(input_dir, "CanopyHeight.tif"),
  human_footprint = file.path(input_dir, "HumanImpact.tif"),
  precipitation   = file.path(input_dir, "Precipitation.tif"),
  temperature     = file.path(input_dir, "TCB.tif"),
  tc_brightness   = file.path(input_dir, "TCG.tif"),
  tc_greenness    = file.path(input_dir, "TCW.tif"),
  tc_wetness      = file.path(input_dir, "Temperature.tif")
)

# Potential diversity input options
use_existing_P_raster <- FALSE
potential_diversity_raster_file <- file.path(input_dir, "potential_diversity_P_100m.tif")
habitat_class_raster_file <- file.path(input_dir, "prehuman_habitat_classes_100m.tif")

# Modelling options
n_background_candidates <- 2000
presence_absence_ratio <- 1
n_folds <- 5
low_conf_auc_threshold <- 0.70

# Spatial thinning distance in metres
thin_distance_m <- 1000
africa_equal_area_crs <- "ESRI:102022"

# Output files
clean_occurrence_output <- file.path(output_dir, "cleaned_occurrences.gpkg")
pa_points_output <- file.path(output_dir, "presence_absence_points.gpkg")
folds_output <- file.path(output_dir, "spatial_block_folds.gpkg")
fold_auc_output <- file.path(output_dir, "fold_level_auc.csv")

D_raster_output <- file.path(output_dir, "D_ensemble_suitability.tif")
P_raster_output <- file.path(output_dir, "P_potential_diversity.tif")
ibii_raster_output <- file.path(output_dir, "iBII_final.tif")
fold_auc_raster_output <- file.path(output_dir, "fold_level_AUC_surface.tif")
low_conf_mask_output <- file.path(output_dir, "iBII_low_confidence_mask_AUC_lt_070.tif")

map_output_png <- file.path(output_dir, "iBII_with_uncertainty_overlay.png")
map_output_pdf <- file.path(output_dir, "iBII_with_uncertainty_overlay.pdf")


############################################################
# 2. Load study boundary
############################################################

if (file.exists(africa_boundary_file)) {
  africa <- st_read(africa_boundary_file, quiet = TRUE)
} else {
  geojson_url <- "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
  world <- st_read(geojson_url, quiet = TRUE)
  africa <- world[world$CONTINENT == "Africa", ]
}

sf_use_s2(FALSE)
africa <- st_make_valid(africa)
africa <- st_union(africa)
sf_use_s2(TRUE)
africa <- st_as_sf(africa)

africa_eq <- st_transform(africa, africa_equal_area_crs)


############################################################
# 3. Load and harmonize predictor rasters (Vectorized Stream)
############################################################

predictors_raw <- rast(predictor_files)
names(predictors_raw) <- names(predictor_files)
template <- predictors_raw[[1]]

aligned_list <- lapply(seq_along(predictor_files), function(i) {
  r <- predictors_raw[[i]]
  if (!compareGeom(r, template, stopOnError = FALSE)) {
    r <- project(r, template)
    r <- resample(r, template, method = "bilinear")
  }
  return(r)
})

predictors <- rast(aligned_list)
names(predictors) <- names(predictor_files)

# Mask predictors to Africa extent
africa_vect <- vect(st_transform(africa, crs(template)))
predictors <- crop(predictors, africa_vect)
predictors <- mask(predictors, africa_vect)

# Calculate global means and SDs
means <- global(predictors, "mean", na.rm = TRUE)$mean
sds   <- global(predictors, "sd", na.rm = TRUE)$sd

# Standardize predictors using terra native streaming
zscore_output_file <- file.path(output_dir, "predictors_zscore.tif")
predictors_z <- (predictors - means) / sds

writeRaster(
  predictors_z, 
  filename = zscore_output_file, 
  overwrite = TRUE, 
  gdal = c("COMPRESS=LZW", "PREDICTOR=2")
)

# Re-read static disk-backed standardized predictors
predictors_z <- rast(zscore_output_file)
names(predictors_z) <- names(predictor_files)


############################################################
# 4. Load and clean occurrence data
############################################################

occ <- read_csv(occurrence_file, show_col_types = FALSE)

if (!all(c("longitude", "latitude") %in% names(occ))) {
  stop("Occurrence CSV must contain columns named 'longitude' and 'latitude'.")
}

occ <- occ %>%
  filter(!is.na(longitude), !is.na(latitude)) %>%
  filter(longitude >= -25, longitude <= 60) %>%
  filter(latitude >= -40, latitude <= 40) %>%
  distinct(longitude, latitude, .keep_all = TRUE)

if ("coordinateUncertaintyInMeters" %in% names(occ)) {
  occ <- occ %>%
    filter(is.na(coordinateUncertaintyInMeters) | coordinateUncertaintyInMeters <= 100)
}

occ_sf <- st_as_sf(occ, coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)
occ_sf <- st_make_valid(occ_sf)
occ_sf <- suppressWarnings(st_intersection(occ_sf, st_transform(africa, 4326)))

# Filter out records outside raster predictors valid mask
occ_pred <- terra::extract(predictors_z, vect(occ_sf))
occ_sf$valid_predictors <- complete.cases(occ_pred[, -1])
occ_sf <- occ_sf %>% filter(valid_predictors)

# Thinning Function
thin_points <- function(points_sf, min_dist) {
  if (nrow(points_sf) <= 1) return(points_sf)
  
  coords <- st_coordinates(points_sf)
  selected <- rep(TRUE, nrow(points_sf))
  
  for (i in seq_len(nrow(points_sf) - 1)) {
    if (!selected[i]) next
    dists <- sqrt((coords[(i + 1):nrow(coords), 1] - coords[i, 1])^2 +
                    (coords[(i + 1):nrow(coords), 2] - coords[i, 2])^2)
    selected[(i + 1):nrow(coords)][dists < min_dist] <- FALSE
  }
  points_sf[selected, ]
}

occ_eq <- st_transform(occ_sf, africa_equal_area_crs)

if ("order" %in% names(occ_eq)) {
  occ_thin_eq <- occ_eq %>%
    group_by(order) %>%
    group_modify(~ thin_points(.x, thin_distance_m)) %>%
    ungroup()
} else {
  occ_thin_eq <- thin_points(occ_eq, thin_distance_m)
}

occ_clean <- st_transform(occ_thin_eq, crs(template))
st_write(occ_clean, clean_occurrence_output, delete_dsn = TRUE, quiet = TRUE)
message("Cleaned occurrence records: ", nrow(occ_clean))


############################################################
# 5. Generate pseudo-absence / background points
############################################################

valid_mask <- !is.na(predictors_z[[1]])

candidate_bg <- spatSample(
  valid_mask,
  size = n_background_candidates,
  method = "random",
  as.points = TRUE,
  na.rm = TRUE,
  values = FALSE
)

candidate_bg_sf <- st_as_sf(candidate_bg)
st_crs(candidate_bg_sf) <- crs(template)

presence_eq <- st_transform(occ_clean, africa_equal_area_crs)
bg_eq <- st_transform(candidate_bg_sf, africa_equal_area_crs)

nearest_idx <- st_nearest_feature(bg_eq, presence_eq)
dist_vec <- st_distance(bg_eq, presence_eq[nearest_idx, ], by_element = TRUE)

bg_eq$min_dist_to_presence <- as.numeric(dist_vec)
bg_eq <- bg_eq %>% filter(min_dist_to_presence >= thin_distance_m)

n_pres <- nrow(occ_clean)
n_abs <- min(nrow(bg_eq), n_pres * presence_absence_ratio)

set.seed(123)
bg_eq <- bg_eq[sample(seq_len(nrow(bg_eq)), n_abs), ]
bg_sf <- st_transform(bg_eq, crs(template))

# Combine Presences and Absences
presence_sf <- occ_clean[, "geometry"] %>% mutate(presence = 1)
absence_sf  <- bg_sf[, "geometry"] %>% mutate(presence = 0)

pa_sf <- rbind(presence_sf, absence_sf)
pa_sf$id <- seq_len(nrow(pa_sf))

pa_values <- terra::extract(predictors_z, vect(pa_sf))
pa_df <- st_drop_geometry(pa_sf) %>%
  bind_cols(pa_values[, -1]) %>%
  filter(complete.cases(.))

pa_sf <- pa_sf[pa_sf$id %in% pa_df$id, ]


############################################################
# 6. Latitudinal block cross-validation (5 Folds)
############################################################

pa_cv <- st_transform(pa_sf, 4326)

ymin <- min(st_coordinates(pa_cv)[, 2])
ymax <- max(st_coordinates(pa_cv)[, 2])

lat_breaks <- seq(ymin, ymax, length.out = n_folds + 1)

pa_cv$fold_id <- cut(
  st_coordinates(pa_cv)[, 2], 
  breaks = lat_breaks, 
  labels = 1:n_folds, 
  include.lowest = TRUE
)
pa_cv$fold_id <- as.integer(as.character(pa_cv$fold_id))

pa_sf <- st_transform(pa_cv, crs(template))
st_write(pa_sf, pa_points_output, delete_dsn = TRUE, quiet = TRUE)

xmin <- min(st_coordinates(pa_cv)[, 1]) - 1
xmax <- max(st_coordinates(pa_cv)[, 1]) + 1

blocks_list <- lapply(1:n_folds, function(k) {
  bbox <- st_bbox(c(
    xmin = xmin, 
    ymin = lat_breaks[k], 
    xmax = xmax, 
    ymax = lat_breaks[k + 1]
  ), crs = st_crs(4326))
  
  poly <- st_as_sfc(bbox)
  st_sf(block_id = k, fold_id = k, geometry = poly)
})

blocks <- do.call(rbind, blocks_list)
blocks <- st_transform(blocks, crs(template))
st_write(blocks, folds_output, delete_dsn = TRUE, quiet = TRUE)

############################################################
# Visualizing Latitudinal Folds with Occurrences Overlayed
############################################################

# 1. Prepare data in WGS84 for plotting
africa_wgs84 <- st_transform(africa, 4326)
blocks_wgs84 <- st_transform(blocks, 4326)
pa_wgs84     <- st_transform(pa_sf, 4326)

# Separate presences and absences for distinct styling
presences_wgs84 <- pa_wgs84 %>% filter(presence == 1)
absences_wgs84  <- pa_wgs84 %>% filter(presence == 0)

# 2. Build map plot
fold_plot <- ggplot() +
  geom_sf(data = africa_wgs84, fill = "gray95", color = "gray70", size = 0.3) +
  geom_sf(data = blocks_wgs84, aes(fill = factor(fold_id)), alpha = 0.4, color = "black", linetype = "dashed", size = 0.5) +
  geom_sf_text(data = blocks_wgs84, aes(label = paste("Fold", fold_id)), color = "gray20", fontface = "bold", size = 4.5) +
  geom_sf(data = absences_wgs84, color = "#d95f02", size = 0.8, alpha = 0.3) +
  geom_sf(data = presences_wgs84, color = "#1b9e77", size = 1.2, alpha = 0.7) +
  scale_fill_brewer(palette = "Set3", name = "Latitudinal Fold") +
  coord_sf(xlim = c(-20, 55), ylim = c(-36, 38), expand = FALSE) +
  labs(
    title = "5-Fold Latitudinal Block Cross-Validation",
    subtitle = "Green: Presence Occurrences | Orange: Pseudo-Absences",
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "gray30"),
    legend.position = "right",
    panel.grid.major = element_line(color = "gray90", linetype = "dotted")
  )

# 3. Save map to disk
ggsave(
  filename = file.path(output_dir, "latitudinal_folds_with_occurrences.png"),
  plot = fold_plot,
  width = 10,
  height = 9,
  dpi = 300
)

print(fold_plot)


############################################################
# 7. Prepare modelling dataframe
############################################################

pa_extract <- terra::extract(predictors_z, vect(pa_sf))

# Explicitly cast to standard data.frame to prevent tibble matrix-indexing errors in maxnet
model_df <- st_drop_geometry(pa_sf) %>%
  select(id, presence, fold_id) %>%
  bind_cols(pa_extract[, -1]) %>%
  filter(complete.cases(.)) %>%
  as.data.frame()

model_df$presence <- as.factor(model_df$presence)
predictor_names <- names(predictors_z)


############################################################
# 8. Helper functions for metrics
############################################################

calc_metrics <- function(obs, pred_prob) {
  obs_numeric <- as.numeric(as.character(obs))
  
  # 1. Safely calculate ROC / AUC
  auc_val <- tryCatch({
    roc_obj <- pROC::roc(
      response = obs_numeric, 
      predictor = pred_prob, 
      levels = c(0, 1),
      direction = "<", # Explicit direction prevents pROC internal subsetting bugs
      quiet = TRUE
    )
    as.numeric(pROC::auc(roc_obj))
  }, error = function(e) {
    NA_real_
  })
  
  # 2. Evaluate thresholds
  thresholds <- seq(0.01, 0.99, by = 0.01)
  
  metric_table <- lapply(thresholds, function(th) {
    pred_class <- ifelse(pred_prob >= th, 1, 0)
    tp <- sum(pred_class == 1 & obs_numeric == 1) 
    tn <- sum(pred_class == 0 & obs_numeric == 0)
    fp <- sum(pred_class == 1 & obs_numeric == 0)
    fn <- sum(pred_class == 0 & obs_numeric == 1)
    
    sens <- ifelse((tp + fn) == 0, NA, tp / (tp + fn))
    spec <- ifelse((tn + fp) == 0, NA, tn / (tn + fp))
    tss  <- sens + spec - 1
    
    data.frame(threshold = th, sensitivity = sens, specificity = spec, tss = tss)
  }) %>% bind_rows()
  
  # Pick best TSS threshold
  best <- metric_table %>% 
    filter(!is.na(tss)) %>% 
    filter(tss == max(tss, na.rm = TRUE)) %>% 
    slice(1)
  
  if (nrow(best) == 0) {
    return(data.frame(
      AUC = NA, TSS = NA, threshold = NA, 
      sensitivity = NA, specificity = NA, kappa = NA
    ))
  }
  
  pred_class_best <- ifelse(pred_prob >= best$threshold, 1, 0)
  
  cm <- caret::confusionMatrix(
    factor(pred_class_best, levels = c(0, 1)),
    factor(obs_numeric, levels = c(0, 1))
  )
  
  data.frame(
    AUC = auc_val,
    TSS = best$tss,
    threshold = best$threshold,
    sensitivity = best$sensitivity,
    specificity = best$specificity,
    kappa = as.numeric(cm$overall["Kappa"])
  )
}

############################################################
# 9. Model fitting & prediction functions (RF, XGBoost, MaxEnt)
############################################################

fit_models <- function(train_df) {
  # Enforce base data.frame
  train_df <- as.data.frame(train_df)
  train_df$presence_num <- as.numeric(as.character(train_df$presence))
  
  X_train <- train_df[, predictor_names, drop = FALSE]
  
  # 1. Random Forest
  rf_mod <- randomForest::randomForest(
    x = X_train, 
    y = train_df$presence, 
    ntree = 500
  )
  
  # 2. XGBoost
  dtrain <- xgboost::xgb.DMatrix(
    data = as.matrix(X_train),
    label = train_df$presence_num
  )
  
  xgb_params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 4,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  xgb_mod <- xgboost::xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = 300,
    verbose = 0
  )
  
  # 3. MaxEnt (maxnet) with automatic class simplification fallback
  maxent_mod <- tryCatch({
    maxnet::maxnet(
      p = train_df$presence_num,
      data = X_train,
      f = maxnet::maxnet.formula(p = train_df$presence_num, data = X_train, classes = "lqph")
    )
  }, error = function(e) {
    message("  [Maxnet Warning] Full 'lqph' features failed on this fold split. Falling back to 'lq' classes.")
    maxnet::maxnet(
      p = train_df$presence_num,
      data = X_train,
      f = maxnet::maxnet.formula(p = train_df$presence_num, data = X_train, classes = "lq")
    )
  })
  
  list(RF = rf_mod, XGBoost = xgb_mod, MaxEnt = maxent_mod)
}

predict_models_df <- function(models, new_df) {
  # Enforce base data.frame
  new_df <- as.data.frame(new_df)
  X_new <- new_df[, predictor_names, drop = FALSE]
  pred <- list()
  
  # RF
  pred$RF <- predict(models$RF, newdata = X_new, type = "prob")[, "1"]
  
  # XGBoost
  dtest <- xgboost::xgb.DMatrix(data = as.matrix(X_new))
  pred$XGBoost <- predict(models$XGBoost, newdata = dtest)
  
  # MaxEnt
  pred$MaxEnt <- as.numeric(predict(models$MaxEnt, X_new, type = "logistic"))
  
  as.data.frame(pred)
}

############################################################
# 10. Spatial block cross-validation execution
############################################################

fold_ids <- sort(unique(model_df$fold_id))
all_fold_metrics <- list()

for (fold in fold_ids) {
  message("Running fold ", fold)
  
  train_df <- model_df %>% filter(fold_id != fold) %>% as.data.frame()
  test_df  <- model_df %>% filter(fold_id == fold) %>% as.data.frame()
  
  models  <- fit_models(train_df)
  pred_df <- predict_models_df(models, test_df)
  
  fold_metrics <- lapply(names(pred_df), function(model_name) {
    m <- calc_metrics(test_df$presence, pred_df[[model_name]])
    m$model <- model_name
    m$fold_id <- fold
    m
  }) %>% bind_rows()
  
  ensemble_prob <- rowMeans(pred_df, na.rm = TRUE)
  ensemble_metrics <- calc_metrics(test_df$presence, ensemble_prob)
  ensemble_metrics$model <- "Ensemble_mean"
  ensemble_metrics$fold_id <- fold
  
  all_fold_metrics[[as.character(fold)]] <- bind_rows(fold_metrics, ensemble_metrics)
}

cv_metrics <- bind_rows(all_fold_metrics)
write_csv(cv_metrics, file.path(output_dir, "cross_validation_metrics_all_models.csv"))

fold_auc <- cv_metrics %>%
  filter(model == "Ensemble_mean") %>%
  select(fold_id, AUC, TSS, sensitivity, specificity, kappa)

write_csv(fold_auc, fold_auc_output)


############################################################
# 11. Train final models on full data
############################################################

final_models <- fit_models(model_df)

model_auc_weights <- cv_metrics %>%
  filter(model %in% c("RF", "XGBoost", "MaxEnt")) %>%
  group_by(model) %>%
  summarize(mean_AUC = mean(AUC, na.rm = TRUE), .groups = "drop") %>%
  mutate(weight = mean_AUC / sum(mean_AUC, na.rm = TRUE))

write_csv(model_auc_weights, file.path(output_dir, "model_auc_weights.csv"))


############################################################
# 12. Chunked Predict suitability rasters to Disk
############################################################

predict_raster_model_chunked <- function(model_name, model_object, predictors_z, output_file) {
  message("Predicting raster in chunks for ", model_name)
  
  out_r <- rast(predictors_z[[1]])
  b <- blocks(predictors_z)
  
  readStart(predictors_z)
  writeStart(out_r, filename = output_file, overwrite = TRUE, gdal = c("COMPRESS=LZW", "PREDICTOR=2"))
  
  for (i in seq_len(b$n)) {
    val_block <- readValues(predictors_z, row = b$row[i], nrows = b$nrows[i], mat = TRUE)
    val_df <- as.data.frame(val_block)
    
    pred_vals <- rep(NA_real_, nrow(val_df))
    valid_idx <- complete.cases(val_df)
    
    if (any(valid_idx)) {
      sub_df <- val_df[valid_idx, , drop = FALSE]
      
      p_res <- switch(
        model_name,
        "RF"      = predict(model_object, newdata = sub_df, type = "prob")[, "1"],
        "XGBoost" = predict(model_object, newdata = xgboost::xgb.DMatrix(data = as.matrix(sub_df))),
        "MaxEnt"  = predict(model_object, sub_df, type = "logistic")
      )
      pred_vals[valid_idx] <- as.numeric(p_res)
    }
    
    writeValues(out_r, pred_vals, b$row[i], b$nrows[i])
  }
  
  readStop(predictors_z)
  writeStop(out_r)
  
  return(output_file)
}

model_pred_files <- list()

for (model_name in names(final_models)) {
  out_file <- file.path(output_dir, paste0("pred_", model_name, ".tif"))
  predict_raster_model_chunked(model_name, final_models[[model_name]], predictors_z, out_file)
  model_pred_files[[model_name]] <- out_file
}

gc()


############################################################
# 13. Chunked AUC-weighted ensemble suitability D
############################################################

weights_named <- model_auc_weights$weight
names(weights_named) <- model_auc_weights$model

pred_stack <- rast(unlist(model_pred_files[names(weights_named)]))

D <- rast(pred_stack[[1]])
b <- blocks(pred_stack)

readStart(pred_stack)
writeStart(D, filename = D_raster_output, overwrite = TRUE, gdal = c("COMPRESS=LZW", "PREDICTOR=2"))

for (i in seq_len(b$n)) {
  val_block <- readValues(pred_stack, row = b$row[i], nrows = b$nrows[i], mat = TRUE)
  
  weighted_vals <- apply(val_block, 1, function(row) {
    if (all(is.na(row))) return(NA_real_)
    sum(row * weights_named, na.rm = TRUE) / sum(weights_named[!is.na(row)])
  })
  
  weighted_vals <- pmax(0, pmin(1, weighted_vals))
  writeValues(D, weighted_vals, b$row[i], b$nrows[i])
}

readStop(pred_stack)
writeStop(D)

D <- rast(D_raster_output)
names(D) <- "D_ensemble_suitability"


############################################################
# 14. Chunked Prepare potential diversity P
############################################################

if (use_existing_P_raster && file.exists(potential_diversity_raster_file)) {
  P_raw <- rast(potential_diversity_raster_file)
  if (!compareGeom(P_raw, D, stopOnError = FALSE)) {
    P_raw <- project(P_raw, D)
    P_raw <- resample(P_raw, D, method = "bilinear")
  }
  P_raw <- crop(P_raw, D)
  P_raw <- mask(P_raw, D)
  
  P <- rast(D)
  b <- blocks(P_raw)
  readStart(P_raw)
  writeStart(P, filename = P_raster_output, overwrite = TRUE, gdal = c("COMPRESS=LZW", "PREDICTOR=2"))
  
  for (i in seq_len(b$n)) {
    v <- readValues(P_raw, row = b$row[i], nrows = b$nrows[i], mat = TRUE)
    v[v <= 0] <- NA
    v <- pmax(0, pmin(1, v))
    writeValues(P, as.numeric(v), b$row[i], b$nrows[i])
  }
  readStop(P_raw)
  writeStop(P)
  
} else {
  if (file.exists(habitat_class_raster_file)) {
    habitat <- rast(habitat_class_raster_file)
    if (!compareGeom(habitat, D, stopOnError = FALSE)) {
      habitat <- project(habitat, D, method = "near")
      habitat <- resample(habitat, D, method = "near")
    }
    habitat <- crop(habitat, D)
    habitat <- mask(habitat, D)
    
    p_lookup <- c("1" = 1.00, "2" = 0.90, "3" = 0.85, "4" = 0.80, "5" = 0.70, "6" = 0.65, "7" = 0.50, "8" = 0.40)
    
    P <- rast(D)
    b <- blocks(habitat)
    readStart(habitat)
    writeStart(P, filename = P_raster_output, overwrite = TRUE, gdal = c("COMPRESS=LZW", "PREDICTOR=2"))
    
    for (i in seq_len(b$n)) {
      v <- readValues(habitat, row = b$row[i], nrows = b$nrows[i], mat = TRUE)
      v_mapped <- p_lookup[as.character(v)]
      writeValues(P, as.numeric(v_mapped), b$row[i], b$nrows[i])
    }
    readStop(habitat)
    writeStop(P)
    
  } else {
    warning("Potential diversity/Habitat inputs not found. Initializing constant P layer = 1.0.")
    P <- rast(D)
    b <- blocks(D)
    readStart(D)
    writeStart(P, filename = P_raster_output, overwrite = TRUE, gdal = c("COMPRESS=LZW", "PREDICTOR=2"))
    
    for (i in seq_len(b$n)) {
      v <- readValues(D, row = b$row[i], nrows = b$nrows[i], mat = TRUE)
      v[!is.na(v)] <- 1.0
      writeValues(P, as.numeric(v), b$row[i], b$nrows[i])
    }
    readStop(D)
    writeStop(P)
  }
}

P <- rast(P_raster_output)
names(P) <- "P_potential_diversity"


############################################################
# 15. Chunked Compute i-BII = D / P
############################################################

ibii_stack <- c(D, P)
ibii <- rast(D)

b <- blocks(ibii_stack)
readStart(ibii_stack)
writeStart(ibii, filename = ibii_raster_output, overwrite = TRUE, gdal = c("COMPRESS=LZW", "PREDICTOR=2"))

for (i in seq_len(b$n)) {
  v <- readValues(ibii_stack, row = b$row[i], nrows = b$nrows[i], mat = TRUE)
  
  d_val <- v[, 1]
  p_val <- v[, 2]
  
  ratio <- d_val / p_val
  ratio <- pmax(0, pmin(1, ratio))
  
  writeValues(ibii, ratio, b$row[i], b$nrows[i])
}

readStop(ibii_stack)
writeStop(ibii)

ibii <- rast(ibii_raster_output)
names(ibii) <- "iBII"


############################################################
# 16. Fold-level AUC surface and low-confidence mask
############################################################

blocks_auc <- blocks %>%
  left_join(fold_auc, by = "fold_id") %>%
  mutate(low_confidence = ifelse(AUC < low_conf_auc_threshold, 1, 0))

st_write(blocks_auc, file.path(output_dir, "spatial_block_folds_with_auc.gpkg"), delete_dsn = TRUE, quiet = TRUE)

blocks_auc_vect <- vect(blocks_auc)

fold_auc_surface <- rasterize(blocks_auc_vect, ibii, field = "AUC", touches = TRUE)
names(fold_auc_surface) <- "fold_level_AUC"
writeRaster(fold_auc_surface, fold_auc_raster_output, overwrite = TRUE, gdal = c("COMPRESS=LZW", "PREDICTOR=2"))

low_conf_mask <- rasterize(blocks_auc_vect, ibii, field = "low_confidence", touches = TRUE)
names(low_conf_mask) <- "low_confidence_AUC_lt_070"
writeRaster(low_conf_mask, low_conf_mask_output, overwrite = TRUE, gdal = c("COMPRESS=LZW", "PREDICTOR=2"))


############################################################
# 17. Export maps (Terra Native Graphics Rendering)
############################################################

png(map_output_png, width = 3000, height = 2500, res = 300)
plot(ibii, main = "Insect-Based Biodiversity Intactness Index (i-BII)", col = rev(terrain.colors(100)))
plot(st_geometry(st_transform(africa, crs(ibii))), add = TRUE, border = "black", lwd = 0.5)
dev.off()

pdf(map_output_pdf, width = 10, height = 8)
plot(ibii, main = "Insect-Based Biodiversity Intactness Index (i-BII)", col = rev(terrain.colors(100)))
plot(st_geometry(st_transform(africa, crs(ibii))), add = TRUE, border = "black", lwd = 0.5)
dev.off()

png(file.path(output_dir, "fold_level_AUC_map.png"), width = 3000, height = 2500, res = 300)
plot(fold_auc_surface, main = "Spatial Block Cross-Validation Fold-Level AUC", col = viridis::viridis(100))
plot(st_geometry(st_transform(africa, crs(ibii))), add = TRUE, border = "black", lwd = 0.5)
dev.off()


############################################################
# 18. Clean temporary files & Export summary table
############################################################

for (f in model_pred_files) {
  if (file.exists(f)) unlink(f)
}

summary_df <- data.frame(
  metric = c("n_clean_occurrences", "n_presence_absence_points", "n_folds", "mean_ensemble_auc", "min_ensemble_auc", "max_ensemble_auc", "low_confidence_threshold"),
  value = c(nrow(occ_clean), nrow(pa_sf), length(fold_ids), mean(fold_auc$AUC, na.rm = TRUE), min(fold_auc$AUC, na.rm = TRUE), max(fold_auc$AUC, na.rm = TRUE), low_conf_auc_threshold)
)

write_csv(summary_df, file.path(output_dir, "ibii_pipeline_summary.csv"))

message("Pipeline completed successfully.")
message("Outputs written to: ", output_dir)