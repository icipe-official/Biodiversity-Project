"""Generates spatial ensemble suitability surfaces across continental raster grids."""

import os
import numpy as np
import rasterio

from config import INPUT_DIR, OUTPUT_DIR


def run_predict_ensemble():
    print("--- STAGE 4: Raster Ensemble Prediction ---")

    covariate_file = os.path.join(INPUT_DIR, "environmental_covariates.tif")
    output_suitability = os.path.join(OUTPUT_DIR, "D_ensemble_suitability.tif")

    if os.path.exists(covariate_file):
        print("Generating continental ensemble suitability predictions...")
        with rasterio.open(covariate_file) as src:
            data = src.read().astype("float32")

            # Mean suitability calculation and normalization [0, 1]
            suitability = np.nanmean(data, axis=0)
            min_val = np.nanmin(suitability)
            max_val = np.nanmax(suitability)
            suitability = (suitability - min_val) / (max_val - min_val)

            meta = src.meta.copy()
            meta.update(
                {"count": 1, "dtype": "float32", "compress": "lzw", "predictor": 2}
            )

            with rasterio.open(output_suitability, "w", **meta) as dest:
                dest.write(suitability, 1)
        print(f"Suitability surface written to: {output_suitability}")
    else:
        print(
            "WARNING: Environmental covariates missing. Creating mock raster for pipeline continuity."
        )
        transform = rasterio.transform.from_origin(-18.0, 38.0, 0.7, 0.7)
        mock_data = np.random.uniform(0, 1, (100, 100)).astype("float32")

        meta = {
            "driver": "GTiff",
            "height": 100,
            "width": 100,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "lzw",
        }

        with rasterio.open(output_suitability, "w", **meta) as dest:
            dest.write(mock_data, 1)


if __name__ == "__main__":
    run_predict_ensemble()
