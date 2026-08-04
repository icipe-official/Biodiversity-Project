"""Occurrence Pre-Processing, 150 km Spatial Grid Thinning, and Dataset Merging."""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from config import AFRICA_EQUAL_AREA_CRS, INPUT_DIR


def run_preprocess():
    print("--- STAGE 0: Occurrence Pre-Processing & Grid Thinning ---")

    shp_file = os.path.join(INPUT_DIR, "Butterfly_Moth.shp")
    file2 = os.path.join(INPUT_DIR, "Butterfly_Moth_combined_thinned_5km.csv")
    out_csv = os.path.join(INPUT_DIR, "Butterfly_Moth_combined_final.csv")

    # 1. Spatial Grid Thinning on Shapefile (150 km grid in ESRI:102022)
    if os.path.exists(shp_file):
        print("Thinning primary shapefile using 150 km grid...")
        pts = gpd.read_file(shp_file)
        pts_projected = pts.to_crs(AFRICA_EQUAL_AREA_CRS)

        bounds = pts_projected.total_bounds  # [minx, miny, maxx, maxy]
        cell_size = 150000.0  # 150 km

        x_coords = np.arange(bounds[0], bounds[2] + cell_size, cell_size)
        y_coords = np.arange(bounds[1], bounds[3] + cell_size, cell_size)

        grid_polys = []
        for x in x_coords:
            for y in y_coords:
                grid_polys.append(box(x, y, x + cell_size, y + cell_size))

        grid_gdf = gpd.GeoDataFrame(
            {"grid_id": range(len(grid_polys))},
            geometry=grid_polys,
            crs=AFRICA_EQUAL_AREA_CRS,
        )

        # Spatial join points to grid cells and take 1 record per grid cell
        joined = gpd.sjoin(pts_projected, grid_gdf, how="inner", predicate="within")
        thinned = joined.groupby("grid_id").first().reset_index()
        thinned = gpd.GeoDataFrame(
            thinned, geometry="geometry", crs=AFRICA_EQUAL_AREA_CRS
        ).to_crs("EPSG:4326")

        df1 = pd.DataFrame(
            {"longitude": thinned.geometry.x, "latitude": thinned.geometry.y}
        )

        print(f"Original shapefile points: {len(pts)}")
        print(f"150 km thinned points: {len(df1)}")
    else:
        sys.exit(f"Input shapefile not found at expected path: {shp_file}")

    # 2. Ingest Secondary 5 km Thinned Dataset
    if os.path.exists(file2):
        print("Merging with secondary 5 km dataset...")
        df2 = pd.read_csv(file2)
        df2.columns = df2.columns.str.lower()
        df2 = df2[["longitude", "latitude"]]
    else:
        print("Secondary dataset not found; proceeding with primary thinned set only.")
        df2 = pd.DataFrame(columns=["longitude", "latitude"])

    # 3. Combine and Deduplicate Records
    df1.columns = df1.columns.str.lower()
    df1 = df1[["longitude", "latitude"]]

    combined = pd.concat([df1, df2], ignore_index=True)
    combined = combined.dropna(subset=["longitude", "latitude"])
    combined = combined.drop_duplicates(subset=["longitude", "latitude"]).reset_index(
        drop=True
    )

    combined.to_csv(out_csv, index=False)

    print(f"Combined unique records successfully written to: {out_csv}")
    print(f"Total Final Occurrences: {len(combined)}")


if __name__ == "__main__":
    run_preprocess()
