# Insect-Based Biodiversity Intactness Index (i-BII) [Python Edition]

A reproducible Python spatial modeling pipeline for mapping continental-scale insect biodiversity intactness across Africa using spatial block cross-validation, potential diversity normalizations, and observational uncertainty masking.

## Pipeline Architecture

The workflow is structured into modular execution stages:

1. **`src/00_preprocess_occurrences.py`**: Performs grid-based spatial thinning on species occurrence shapefiles (`ESRI:102022`) and merges records against secondary CSV datasets.
2. **`src/01_data_prep.py`**: Extracts environmental spatial covariates for presence/pseudo-absence sites.
3. **`src/02_block_cv.py`**: Constructs spatial block cross-validation folds to prevent spatial autocorrelation bias.
4. **`src/03_model_training.py`**: Fits ensemble distribution models and calculates cross-validation fold evaluation metrics (AUC).
5. **`src/04_predict_ensemble.py`**: Predicts continental-scale ensemble suitability surfaces ($D$).
6. **`src/05_compute_ibii_exports.py`**: Aligns potential vegetation baselines ($P$), computes the Intactness Ratio ($D/P \in [0,1]$), maps observational uncertainty surfaces, and filters unreliable locations ($> 0.95$).

---

## Installation & Environment Setup

### Option 1: Docker Container (Recommended)

Build the image locally with GDAL, GEOS, and PROJ pre-installed:

```bash
docker build -t ibii-python-pipeline .
```

Execute the pipeline mounting local `data/` directories:

```bash
docker run --rm -v $(pwd)/data:/app/data ibii-python-pipeline --step all
```

### Option 2: Local Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Execution Guide

Run specific stages individually or execute the complete pipeline using `main.py`:

```bash
# Stage 0: Thin and pre-process raw occurrences
python main.py --step preprocess

# Stage 5: Calculate Intactness Ratio & Observational Uncertainty
python main.py --step compute

# Complete End-to-End Execution
python main.py --step all
```

---

## Generated Output Products

All final output products are written to `data/output/`:

* `intactness_ratio.tif`: Bounded Intactness Index ($D / P \in [0, 1]$).
* `observational_support.tif`: Smoothed spatial sampling support surface (25 km neighborhood).
* `iBII_observational_uncertainty.tif`: Observational uncertainty surface ($| \text{Intactness} - \text{Support} |$).
* `iBII_reliable.tif`: Final Intactness surface with highly uncertain locations ($> 0.95$) masked out.
* `iBII_uncertainty_classes.tif`: 5-tier reclassified uncertainty layer.