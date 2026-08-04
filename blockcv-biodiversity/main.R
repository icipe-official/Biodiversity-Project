#!/usr/bin/env Rscript
# main.R
# Command Line Interface (CLI) Execution Driver

suppressPackageStartupMessages(library(optparse))

option_list <- list(
  make_option(c("-s", "--step"), type="character", default="all",
              help="Pipeline step to run: preprocess, prep, block_cv, train, predict, compute, or all [default= %default]", 
              metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

run_step <- function(step_name, script_path) {
  message(sprintf("Running step: [%s]", step_name))
  source(script_path, local = new.env())
}

if (opt$step %in% c("preprocess", "all")) run_step("Pre-process Occurrences", "src/00_preprocess_occurrences.R")
if (opt$step %in% c("prep", "all"))       run_step("Data Preparation", "src/01_data_prep.R")
if (opt$step %in% c("block_cv", "all"))   run_step("Block Cross-Validation", "src/02_block_cv.R")
if (opt$step %in% c("train", "all"))      run_step("Model Training", "src/03_model_training.R")
if (opt$step %in% c("predict", "all"))    run_step("Raster Ensemble Prediction", "src/04_predict_ensemble.R")
if (opt$step %in% c("compute", "all"))    run_step("Compute i-BII Exports", "src/05_compute_ibii_exports.R")

message("Pipeline execution finished successfully.")