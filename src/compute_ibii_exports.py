"""Calculates Intactness Ratio, Observational Uncertainty, Masking & Visual Exporting."""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.rasterize import rasterize
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from config import INPUT_DIR, OUTPUT_DIR, AGGREGATION_FACTOR, UNCERTAINTY_THRESHOLD, LOW_CONF_AUC_THRESHOLD


def run_compute_ibii():
    print("--- STAGE 5: Compute Intactness, Observational Uncertainty & Map Exports ---")

    # ==========================================================================
    # SECTION 1: LOAD SUITABILITY & CALCULATE INTACTNESS RATIO
    # ==========================================================================
    ibii_reliable_file = os.path.join(OUTPUT_DIR, "D_ensemble_suitability.tif")
    if not os.path.exists(ibii_reliable_file):
        sys.exit(f"Ensemble suitability raster not found at: {ibii_reliable_file}")

    with rasterio.open(ibii_reliable_file) as src_d:
        d_val = src_d.read(1).astype("float32")
        meta = src_d.meta.copy()

    potential_layer_file = os.path.join(INPUT_DIR, "pnv_lvl1_004_reclass.tif")
    
    if os.path.exists(potential_layer_file):
        print("Aligning potential layer geometry...")
        with rasterio.open(potential_layer_file) as src_p:
            p_val = src_p.read(
                1,
                out_shape=(meta["height"], meta["width"]),
                resampling=Resampling.nearest
            ).astype("float32")
    else:
        print("Potential layer file not found. Initializing constant baseline P = 1.0.")
        p_val = np.ones_like(d_val, dtype="float32")

    print("Calculating intactness ratio...")
    p_val[p_val <= 0] = np.nan
    ratio = d_val / p_val
    ratio = np.clip(ratio, 0.0, 1.0)

    intactness_output_file = os.path.join(OUTPUT_DIR, "intactness_ratio.tif")
    meta.update({"dtype": "float32", "compress": "lzw", "predictor": 2})

    with rasterio.open(intactness_output_file, "w", **meta) as dest:
        dest.write(ratio, 1)

    print(f"Intactness layer successfully written to: {intactness_output_file}")

    # ==========================================================================
    # SECTION 2: OBSERVATIONAL UNCERTAINTY MAPPING
    # ==========================================================================
    print("--- Computing Observational Uncertainty Surface ---")
    occ_file = os.path.join(INPUT_DIR, "Butterfly_Moth_combined_final.csv")
    if not os.path.exists(occ_file):
        sys.exit(f"Cleaned occurrence dataset not found at: {occ_file}")

    occ_df = pd.read_csv(occ_file)
    coords = list(zip(occ_df["longitude"], occ_df["latitude"]))

    print("Rasterizing occurrence records...")
    with rasterio.open(intactness_output_file) as src:
        shapes = [(geometry, 1) for geometry in gpd.points_from_xy(occ_df.longitude, occ_df.latitude)]
        density = rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            merge_alg=rasterio.enums.MergeAlg.add,
            dtype="float32"
        )

        print("Creating observational support surface...")
        # Smooth occurrence density via gaussian kernel representing neighborhood aggregation
        support = gaussian_filter(density, sigma=AGGREGATION_FACTOR)
        max_support = np.nanmax(support)
        if max_support == 0 or np.isnan(max_support):
            max_support = 1.0

        support = np.clip(support / max_support, 0.0, 1.0)

        support_file = os.path.join(OUTPUT_DIR, "observational_support.tif")
        with rasterio.open(support_file, "w", **meta) as dest:
            dest.write(support, 1)

        print("Calculating uncertainty layer...")
        uncertainty = np.abs(ratio - support)
        uncertainty = np.clip(uncertainty, 0.0, 1.0)

        uncertainty_file = os.path.join(OUTPUT_DIR, "iBII_observational_uncertainty.tif")
        with rasterio.open(uncertainty_file, "w", **meta) as dest:
            dest.write(uncertainty, 1)

        print("Masking highly uncertain regions...")
        reliable_ibii = np.where(uncertainty <= UNCERTAINTY_THRESHOLD, ratio, np.nan)
        reliable_file = os.path.join(OUTPUT_DIR, "iBII_reliable.tif")
        with rasterio.open(reliable_file, "w", **meta) as dest:
            dest.write(reliable_ibii, 1)

        print(f"Uncertainty Summary -> Min: {np.nanmin(uncertainty):.3f}, Mean: {np.nanmean(uncertainty):.3f}, Max: {np.nanmax(uncertainty):.3f}")

        # Classify into 5 discrete tiers
        uncertainty_classes = np.digitize(uncertainty, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        with rasterio.open(os.path.join(OUTPUT_DIR, "iBII_uncertainty_classes.tif"), "w", **meta) as dest:
            dest.write(uncertainty_classes.astype("int32"), 1)

    # ==========================================================================
    # SECTION 3: SPATIAL BLOCK CV MASKING & EXPORTS
    # ==========================================================================
    blocks_sf_file = os.path.join(OUTPUT_DIR, "spatial_block_folds.gpkg")
    fold_auc_file = os.path.join(OUTPUT_DIR, "fold_level_auc.csv")

    if os.path.exists(blocks_sf_file) and os.path.exists(fold_auc_file):
        blocks_gdf = gpd.read_file(blocks_sf_file)
        fold_auc = pd.read_csv(fold_auc_file)

        merged_blocks = blocks_gdf.merge(fold_auc, on="fold_id")
        merged_blocks["low_confidence"] = (merged_blocks["AUC"] < LOW_CONF_AUC_THRESHOLD).astype(int)

        shapes_auc = [(geom, val) for geom, val in zip(merged_blocks.geometry, merged_blocks["AUC"])]
        shapes_low = [(geom, val) for geom, val in zip(merged_blocks.geometry, merged_blocks["low_confidence"])]

        auc_surface = rasterize(shapes=shapes_auc, out_shape=(meta["height"], meta["width"]), transform=meta["transform"], fill=0, dtype="float32")
        low_conf_mask = rasterize(shapes=shapes_low, out_shape=(meta["height"], meta["width"]), transform=meta["transform"], fill=0, dtype="int32")

        with rasterio.open(os.path.join(OUTPUT_DIR, "fold_level_AUC_surface.tif"), "w", **meta) as dest:
            dest.write(auc_surface, 1)

        meta_int = meta.copy()
        meta_int.update({"dtype": "int32"})
        with rasterio.open(os.path.join(OUTPUT_DIR, "iBII_low_confidence_mask_AUC_lt_070.tif"), "w", **meta_int) as dest:
            dest.write(low_conf_mask, 1)

    africa_boundary_file = os.path.join(INPUT_DIR, "africa_boundary.gpkg")
    if os.path.exists(africa_boundary_file):
        africa = gpd.read_file(africa_boundary_file)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(reliable_ibii, cmap="terrain_r", extent=[meta["transform"][0], meta["transform"][0] + meta["transform"][1]*meta["width"], meta["transform"][5] + meta["transform"][4]*meta["height"], meta["transform"][5]])
        africa.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5)
        plt.colorbar(im, ax=ax, label="Reliable iBII")
        plt.title("Reliable Insect Biodiversity Intactness Index (i-BII)")
        plt.savefig(os.path.join(OUTPUT_DIR, "iBII_reliable_map.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print("----------------------------------------")
    print("Stage 5 Execution Summary")
    print("----------------------------------------")
    print(f"Intactness layer: {intactness_output_file}")
    print(f"Observational Support: {support_file}")
    print(f"Observational Uncertainty: {uncertainty_file}")
    print(f"Reliable iBII Layer: {reliable_file}")
    print("----------------------------------------")


if __name__ == "__main__":
    run_compute_ibii()