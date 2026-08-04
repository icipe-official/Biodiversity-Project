"""Prepares environmental predictors, background points, and extracts modeling tables."""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from config import INPUT_DIR, OUTPUT_DIR
from src.preprocess_occurrences import run_preprocess


def run_data_prep():
    occ_file = os.path.join(INPUT_DIR, "Butterfly_Moth_combined_final.csv")
    if not os.path.exists(occ_file):
        print(
            "Combined occurrence file not found. Running pre-processing stage first..."
        )
        run_preprocess()

    print("--- STAGE 1: Data Preparation ---")

    covariate_file = os.path.join(INPUT_DIR, "environmental_covariates.tif")
    occ_df = pd.read_csv(occ_file)

    if os.path.exists(covariate_file):
        print("Extracting environmental values at presence locations...")
        with rasterio.open(covariate_file) as src:
            coords = list(zip(occ_df["longitude"], occ_df["latitude"]))
            extracted_vals = [val for val in src.sample(coords)]

            cov_data = pd.DataFrame(
                extracted_vals, columns=[f"cov_{i + 1}" for i in range(src.count)]
            )
            presence_data = pd.concat([occ_df, cov_data], axis=1)
            presence_data["presence"] = 1

            print("Sampling pseudo-absence background points...")
            np.random.seed(42)
            n_bg = len(occ_df) * 2

            # Read raster mask to select valid non-NA background locations
            band1 = src.read(1)
            valid_indices = (
                np.argwhere(band1 != src.nodata)
                if src.nodata is not None
                else np.argwhere(~np.isnan(band1))
            )

            chosen_idx = valid_indices[
                np.random.choice(len(valid_indices), n_bg, replace=False)
            ]
            bg_coords = [src.xy(r, c) for r, c in chosen_idx]

            bg_extracted = [val for val in src.sample(bg_coords)]
            bg_df = pd.DataFrame(
                {
                    "longitude": [c[0] for c in bg_coords],
                    "latitude": [c[1] for c in bg_coords],
                }
            )
            bg_cov_data = pd.DataFrame(
                bg_extracted, columns=[f"cov_{i + 1}" for i in range(src.count)]
            )
            bg_data = pd.concat([bg_df, bg_cov_data], axis=1)
            bg_data["presence"] = 0

            full_dataset = pd.concat([presence_data, bg_data], ignore_index=True)
            out_file = os.path.join(OUTPUT_DIR, "model_training_dataset.csv")
            full_dataset.to_csv(out_file, index=False)
            print(f"Model training dataset generated successfully at: {out_file}")
    else:
        print(
            f"WARNING: Environmental covariates stack not found at '{covariate_file}'. Skipping extraction step."
        )


if __name__ == "__main__":
    run_data_prep()
