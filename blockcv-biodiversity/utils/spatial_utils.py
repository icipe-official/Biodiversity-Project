"""Utility functions for spatial raster operations and data handling."""

from pathlib import Path
from typing import Union
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask


def apply_spatial_mask(
    input_raster_path: Union[str, Path],
    mask_vector: gpd.GeoDataFrame,
    output_path: Union[str, Path],
) -> None:
    """Apply spatial vector mask to raster safely and write to disk with LZW compression."""
    with rasterio.open(input_raster_path) as src:
        # Transform mask vector to raster CRS if necessary
        if mask_vector.crs != src.crs:
            mask_vector = mask_vector.to_crs(src.crs)

        shapes = [geom for geom in mask_vector.geometry]
        out_image, out_transform = mask(src, shapes, crop=True)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "predictor": 2,
            }
        )

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)


def standardize_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize coordinate column names in a pandas DataFrame."""
    df.columns = df.columns.str.lower()
    if "lon" in df.columns:
        df = df.rename(columns={"lon": "longitude"})
    if "lat" in df.columns:
        df = df.rename(columns={"lat": "latitude"})
    return df
