"""Spatial Block Cross-Validation Fold Allocation."""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from config import AFRICA_EQUAL_AREA_CRS, OUTPUT_DIR


def run_block_cv():
    print("--- STAGE 2: Spatial Block Cross-Validation Setup ---")

    dataset_file = os.path.join(OUTPUT_DIR, "model_training_dataset.csv")

    if os.path.exists(dataset_file):
        df = pd.read_csv(dataset_file)
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
        )
        gdf_proj = gdf.to_crs(AFRICA_EQUAL_AREA_CRS)

        print("Constructing spatial block cross-validation folds...")
        bounds = gdf_proj.total_bounds
        block_size = 250000.0  # 250 km block size

        x_coords = np.arange(bounds[0], bounds[2] + block_size, block_size)
        y_coords = np.arange(bounds[1], bounds[3] + block_size, block_size)

        blocks = []
        for x in x_coords:
            for y in y_coords:
                blocks.append(box(x, y, x + block_size, y + block_size))

        blocks_gdf = gpd.GeoDataFrame(
            {"block_id": range(len(blocks))}, geometry=blocks, crs=AFRICA_EQUAL_AREA_CRS
        )

        # Assign k=5 random folds across blocks
        np.random.seed(42)
        unique_blocks = blocks_gdf["block_id"].values
        fold_assignments = np.random.choice(range(1, 6), size=len(unique_blocks))
        blocks_gdf["fold_id"] = fold_assignments

        joined = gpd.sjoin(
            gdf_proj,
            blocks_gdf[["fold_id", "geometry"]],
            how="inner",
            predicate="within",
        )

        # Save output datasets
        out_df = pd.DataFrame(joined.drop(columns="geometry"))
        out_df.to_csv(
            os.path.join(OUTPUT_DIR, "model_dataset_with_folds.csv"), index=False
        )

        blocks_gdf.to_crs("EPSG:4326").to_file(
            os.path.join(OUTPUT_DIR, "spatial_block_folds.gpkg"), driver="GPKG"
        )

        print("Spatial block CV partitioning completed successfully.")
    else:
        print("WARNING: Training dataset not found. Skipping spatial block generation.")


if __name__ == "__main__":
    run_block_cv()
