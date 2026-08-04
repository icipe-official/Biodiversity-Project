# Insect-Based Biodiversity Intactness Index (i-BII)

A reproducible spatial modeling pipeline for mapping continental-scale insect biodiversity intactness across Africa using spatial block cross-validation, potential diversity normalizations, and observational uncertainty masking.

## Pipeline Architecture

The workflow is structured into modular execution stages:

1. **`src/00_preprocess_occurrences.R`**: Performs 150 km grid-based spatial thinning on species occurrence shapefiles (`ESRI:102022`) and merges records against secondary 5 km CSV datasets.
2. **`src/01_data_prep.R`**: Extracts environmental spatial covariates for presence/pseudo-absence sites.
3. **`src/02_block_cv.R`**: Constructs spatial block cross-validation folds (`blockCV`) to prevent spatial autocorrelation bias.
4. **`src/03_model_training.R`**: Fits ensemble distribution models and calculates cross-validation fold evaluation metrics (AUC).
5. **`src/04_predict_ensemble.R`**: Predicts continental-scale ensemble suitability surfaces ($D$).
6. **`src/05_compute_ibii_exports.R`**: Aligns potential vegetation baselines ($P$), computes the Intactness Ratio ($D/P \in [0,1]$), maps observational uncertainty surfaces, and filters unreliable locations ($> 0.95$).

---

## Installation & Environment Setup

### Option 1: Docker Container (Recommended)

Build the image locally with GDAL, GEOS, and PROJ pre-installed:

```bash
docker build -t ibii-pipeline .
```

Execute the pipeline mounting local `data/` directories:

```bash
docker run --rm -v $(pwd)/data:/app/data ibii-pipeline --step all
```

### Option 2: Local R Installation

Install required R packages:

```R
install.packages(c("terra", "sf", "dplyr", "readr", "blockCV", "optparse", "viridis"))
```

---

## Execution Guide

Run specific stages individually or execute the complete pipeline using `main.R`:

```bash
# Stage 0: Thin and pre-process raw occurrences
Rscript main.R --step preprocess

# Stage 5: Calculate Intactness Ratio & Observational Uncertainty
Rscript main.R --step compute

# Complete End-to-End Execution
Rscript main.R --step all
```

---

## Generated Output Products

All final output products are compressed GTiff rasters written to `data/output/`:

* `intactness_ratio.tif`: Bounded Intactness Index ($D / P \in [0, 1]$).
* `observational_support.tif`: Smoothed spatial sampling support surface (25 km neighborhood).
* `iBII_observational_uncertainty.tif`: Observational uncertainty surface ($| \text{Intactness} - \text{Support} |$).
* `iBII_reliable.tif`: Final Intactness surface with highly uncertain locations ($> 0.95$) masked out.
* `iBII_uncertainty_classes.tif`: 5-tier reclassified uncertainty layer.