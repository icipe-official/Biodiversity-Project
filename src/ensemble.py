#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
AFRICA INSECT SPECIES DISTRIBUTION MODELLING -- 7-ALGORITHM ENSEMBLE
===============================================================================
Algorithms : BRT, GAM, GLM, Maxent, SVM, XGBoost, Random Forest
Workflow   : presence-only records + random background (pseudo-absences)
             -> spatial-block cross-validation -> held-out predictions
             -> performance-weighted ensemble -> suitability rasters
             -> reproducibility report

Runs as a normal script (python sdm_ensemble_7models.py) or pasted into one
Jupyter cell (the last line calls main() in both cases).

Install once:
    pip install numpy pandas scipy scikit-learn joblib rasterio geopandas \
                shapely pyproj xgboost pygam elapid

Every output is written to disk as soon as it is ready.
-------------------------------------------------------------------------------
"""

# SDM_ENSEMBLE_SCRIPT_MARKER  (used to locate this cell's source in a notebook)
from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import importlib.metadata as md
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from sklearn.base import BaseEstimator, TransformerMixin
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"

# =============================================================================
# 1. CONFIGURATION  (edit here, or call main(overrides={...}) from a notebook)
# =============================================================================
CONFIG = dict(
    # ---- inputs / outputs ---------------------------------------------------
    OCCURRENCE_CSV="/home/vincent/Development/BIODIVERISTY_MS_R4/Biodiversity-Project/data/input/CSV/combined_data_landcover_settlement_filtered_20km_thinned.csv",
    RASTER_DIR="/home/vincent/Development/BIODIVERISTY_MS_R4/inputs/rasters/TCI",
    OUTPUT_DIR="/home/vincent/Development/BIODIVERISTY_MS_R4/Biodiversity-Project/data/output",
    AOI_FILE="/home/vincent/Development/BIODIVERISTY_MS_R4/Biodiversity-Project/data/input/shapefiles/Africa_afr_g2014_2013_0.shp",
    RUN_LABEL="ensemble_v1",
    # ---- occurrence file handling ------------------------------------------
    # Column that flags presence (1) vs pseudo-absence (0) inside the input CSV.
    # None = auto-detect (presence/occurrence/pa/response/label). If a binary
    # column is found only the rows == 1 are used as presences; background
    # points are always freshly sampled by this script.
    PRESENCE_COLUMN=None,
    # ---- sampling / validation ---------------------------------------------
    N_BACKGROUND=2000,
    K_FOLDS=5,
    BLOCK_SIZE_M=100_000,  # spatial block size (EPSG:6933, metres)
    COR_THRESHOLD=0.90,  # |r| above which a predictor is dropped
    SEED=21,
    # ---- models -------------------------------------------------------------
    MODELS=["BRT", "GAM", "GLM", "MAXENT", "SVM", "XGB", "RF"],
    N_JOBS=-1,  # -1 = all CPU cores
    TUNE=False,  # quick randomized search for RF / XGB / BRT
    TUNE_ITER=12,
    HP={},  # per-model overrides, e.g. {"RF": {"n_estimators": 1000}}
    # ---- ensemble -----------------------------------------------------------
    WEIGHT_METHOD="auc",  # "auc" | "tss" | "boyce" | "equal"
    WEIGHT_POWER=2.0,  # raw weight = max(metric - baseline, 0) ** power
    MIN_MEAN_AUC=0.55,  # models below this are excluded from the ensemble
    # ---- raster prediction --------------------------------------------------
    CHUNK_CELLS=600_000,  # cells per prediction block (rounded to 256-row tiles)
    WRITE_MODEL_RASTERS=True,  # also write one suitability raster per model
    RESAMPLING="bilinear",  # used only when a predictor must be warped
    NODATA=-9999.0,
    # ---- extra outputs ------------------------------------------------------
    SAVE_PER_FOLD_CSVS=True,
    # ---- reproducibility report --------------------------------------------
    CODE_REPO_URL="",  # e.g. "https://github.com/<user>/<repo>"
    CODE_ARCHIVE_DOI="",  # e.g. "10.5281/zenodo.1234567"
    CODE_ARCHIVE_FILE="",  # optional local zip/tar of the code
    CODE_ARCHIVE_SHA256="",  # optional expected checksum of that archive
    VERIFY_ONLINE=True,  # try to resolve repo URL / DOI (needs internet)
    HASH_FULL_MAX_MB=256,  # files above this get a partial (head+tail) hash
)

DEFAULT_HP = {
    "GLM": dict(C=1.0),
    "GAM": dict(n_splines=8, lam=0.6, max_iter=100),
    "SVM": dict(mode="nystroem", C=1.0, gamma=None, n_components=300),
    "MAXENT": dict(
        feature_types=["linear", "quadratic", "hinge"],
        beta_multiplier=1.5,
        n_hinge_features=10,
    ),
    "XGB": dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
    ),
    "RF": dict(n_estimators=500, max_features="sqrt", min_samples_leaf=3),
    # BRT engine: "hist" (fast, histogram boosting) or "classic" (stochastic GBM)
    "BRT": dict(
        engine="hist",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=10,
        subsample=0.5,
    ),
}

TUNE_GRIDS = {
    "RF": dict(
        n_estimators=[300, 500, 800],
        max_features=["sqrt", 0.3, 0.5],
        min_samples_leaf=[1, 3, 5, 10],
    ),
    "XGB": dict(
        learning_rate=[0.02, 0.05, 0.1],
        max_depth=[2, 3, 4, 6],
        n_estimators=[200, 400, 700],
        subsample=[0.6, 0.8, 1.0],
        colsample_bytree=[0.6, 0.8, 1.0],
        min_child_weight=[1, 3, 5],
    ),
    "BRT": dict(
        learning_rate=[0.02, 0.05, 0.1],
        max_iter=[200, 400, 700],
        max_depth=[2, 3, 4],
        min_samples_leaf=[5, 10, 20],
    ),
}

# ---- optional / heavy imports (reported clearly if missing) -------------------
_MISSING = []


def _try_import(name):
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001
        _MISSING.append(name)
        return None


rasterio = _try_import("rasterio")
gpd = _try_import("geopandas")
shapely = _try_import("shapely")
pyproj = _try_import("pyproj")
joblib = _try_import("joblib")
xgb = _try_import("xgboost")
pygam = _try_import("pygam")
ela = _try_import("elapid")
sklearn = _try_import("sklearn")
scipy_stats = _try_import("scipy.stats")

if rasterio is not None:
    import rasterio.features
    import rasterio.windows
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

LOGGER = logging.getLogger("sdm_ensemble")


def log(msg):
    LOGGER.info(msg)


def setup_logging(out_dir):
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(Path(out_dir) / "run_log.txt", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    LOGGER.addHandler(sh)
    LOGGER.addHandler(fh)


TIMINGS = []


@contextmanager
def stage(name, out_dir=None):
    t0 = time.time()
    log(f"--- {name}: start")
    yield
    dt_s = time.time() - t0
    TIMINGS.append(dict(stage=name, seconds=round(dt_s, 2)))
    log(f"--- {name}: done in {dt_s:,.1f}s")
    if out_dir is not None:
        pd.DataFrame(TIMINGS).to_csv(Path(out_dir) / "stage_timings.csv", index=False)


def save_csv(df, path, what=None):
    """Atomic CSV write; logs row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    log(f"saved {what or path.name}: {len(df):,} rows -> {path}")


def save_json(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


# =============================================================================
# 2. RASTER STACK (lazy, aligned to the reference grid, thread-parallel reads)
# =============================================================================
class Grid:
    def __init__(self, crs, transform, width, height):
        self.crs, self.transform, self.width, self.height = (
            crs,
            transform,
            width,
            height,
        )


def build_grid(ref, aoi_geom):
    """Reference grid cropped to the AOI bounding box (pixel-aligned to ref)."""
    minx, miny, maxx, maxy = aoi_geom.bounds
    w = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=ref.transform)
    c0 = max(0, int(math.floor(w.col_off)))
    r0 = max(0, int(math.floor(w.row_off)))
    c1 = min(ref.width, int(math.ceil(w.col_off + w.width)))
    r1 = min(ref.height, int(math.ceil(w.row_off + w.height)))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("The AOI does not overlap the reference raster.")
    win = Window(c0, r0, c1 - c0, r1 - r0)
    return Grid(ref.crs, ref.window_transform(win), win.width, win.height)


class Layer:
    """One predictor, exposed on the common grid."""

    def __init__(self, path, name, grid, resampling):
        self.path, self.name, self.grid = str(path), name, grid
        self.src = rasterio.open(path)
        s = self.src
        if s.count != 1:
            raise ValueError(f"Predictor must have a single band: {path}")
        if s.crs is None:
            raise ValueError(f"Predictor has no CRS: {path}")
        self.mode, self.off, self.vrt = "warp", (0, 0), None
        t, g = s.transform, grid.transform
        if (
            s.crs == grid.crs
            and math.isclose(t.a, g.a, rel_tol=1e-6)
            and math.isclose(t.e, g.e, rel_tol=1e-6)
            and t.b == 0
            and t.d == 0
        ):
            c0, r0 = (g.c - t.c) / t.a, (g.f - t.f) / t.e
            ci, ri = int(round(c0)), int(round(r0))
            if (
                abs(c0 - ci) < 1e-3
                and abs(r0 - ri) < 1e-3
                and ci >= 0
                and ri >= 0
                and ci + grid.width <= s.width
                and ri + grid.height <= s.height
            ):
                self.mode, self.off = "direct", (ci, ri)
        if self.mode == "warp":
            self.vrt = WarpedVRT(
                s,
                crs=grid.crs,
                transform=grid.transform,
                width=grid.width,
                height=grid.height,
                resampling=resampling,
                src_nodata=s.nodata,
                nodata=float("nan"),
                dtype="float32",
            )

    def read(self, win):
        if self.mode == "direct":
            w = Window(
                win.col_off + self.off[0],
                win.row_off + self.off[1],
                win.width,
                win.height,
            )
            a = self.src.read(1, window=w, masked=True)
        else:
            a = self.vrt.read(1, window=win, masked=True)
        a = np.ma.filled(a.astype("float32"), np.nan)
        a[~np.isfinite(a)] = np.nan
        a[np.abs(a) > 1e30] = np.nan
        return a

    def sample(self, rows, cols):
        out = np.full(len(rows), np.nan, dtype="float32")
        for i, (r, c) in enumerate(zip(rows, cols)):
            out[i] = self.read(Window(int(c), int(r), 1, 1))[0, 0]
        return out

    def close(self):
        for d in (self.vrt, self.src):
            try:
                if d is not None:
                    d.close()
            except Exception:  # noqa: BLE001
                pass


class PredictorStack:
    def __init__(self, files, names, grid, resampling, n_threads):
        self.grid = grid
        self.layers = [Layer(f, n, grid, resampling) for f, n in zip(files, names)]
        self.names = list(names)
        self.pool = ThreadPoolExecutor(
            max_workers=max(1, min(n_threads, len(self.layers)))
        )

    def subset(self, names):
        idx = [self.names.index(n) for n in names]
        return [self.layers[i] for i in idx]

    def sample_cells(self, rows, cols, names=None):
        layers = self.layers if names is None else self.subset(names)
        cols_out = list(self.pool.map(lambda L: L.sample(rows, cols), layers))
        return np.column_stack(cols_out) if cols_out else np.empty((len(rows), 0))

    def read_window(self, win, names):
        arrs = list(self.pool.map(lambda L: L.read(win), self.subset(names)))
        return np.stack(arrs, axis=0)

    def close(self):
        self.pool.shutdown(wait=False)
        for L in self.layers:
            L.close()


def cell_index(grid, x, y):
    t = grid.transform
    cols = np.floor((x - t.c) / t.a).astype("int64")
    rows = np.floor((y - t.f) / t.e).astype("int64")
    ok = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    return rows, cols, ok


def cell_centres(grid, rows, cols):
    t = grid.transform
    return t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e


def clean_name(stem, used):
    n = re.sub(r"\W+", "_", stem).strip("_") or "layer"
    if n[0].isdigit():
        n = "v_" + n
    base, k = n, 1
    while n in used:
        k += 1
        n = f"{base}_{k}"
    used.add(n)
    return n


# =============================================================================
# 3. OCCURRENCES AND BACKGROUND
# =============================================================================
def _find_col(cands, lowered):
    for c in cands:
        if c in lowered:
            return lowered[c]
    return None


def load_presences(cfg, stack, aoi_geom, out_dir):
    csv = cfg["OCCURRENCE_CSV"]
    occ = pd.read_csv(csv, low_memory=False)
    if occ.shape[1] == 1 and "\t" in str(occ.columns[0]):
        occ = pd.read_csv(csv, sep="\t", low_memory=False)
    steps = [("rows in input file", len(occ))]
    lowered = {str(c).strip().lower(): c for c in occ.columns}

    # optional presence / pseudo-absence flag inside the input file
    pcol = cfg.get("PRESENCE_COLUMN")
    if pcol is None:
        pcol = _find_col(
            [
                "presence",
                "occurrence",
                "occurrencestatus",
                "pa",
                "response",
                "label",
                "pres",
            ],
            lowered,
        )
    if pcol is not None and pcol in occ.columns:
        vals = pd.to_numeric(occ[pcol], errors="coerce")
        if set(vals.dropna().unique()) <= {0, 1}:
            n_zero = int((vals == 0).sum())
            occ = occ[vals == 1]
            log(
                f"Column '{pcol}' flags presence/absence: kept {len(occ):,} presences, "
                f"ignored {n_zero:,} rows flagged 0 (background is re-sampled below)."
            )
            steps.append((f"rows flagged presence in '{pcol}'", len(occ)))

    lon_c = _find_col(["decimallongitude", "longitude", "lon", "long", "x"], lowered)
    lat_c = _find_col(["decimallatitude", "latitude", "lat", "y"], lowered)
    if lon_c is None or lat_c is None:
        raise ValueError(f"Cannot find lon/lat columns. Available: {list(occ.columns)}")
    occ = occ.copy()
    occ["longitude"] = pd.to_numeric(occ[lon_c], errors="coerce")
    occ["latitude"] = pd.to_numeric(occ[lat_c], errors="coerce")
    occ = occ[
        np.isfinite(occ.longitude)
        & np.isfinite(occ.latitude)
        & occ.longitude.between(-180, 180)
        & occ.latitude.between(-90, 90)
    ]
    steps.append(("valid lon/lat", len(occ)))
    occ = occ.drop_duplicates(["longitude", "latitude"])
    steps.append(("unique coordinates", len(occ)))

    tr = pyproj.Transformer.from_crs(4326, stack.grid.crs.to_wkt(), always_xy=True)
    x, y = tr.transform(occ.longitude.to_numpy(), occ.latitude.to_numpy())
    x, y = np.asarray(x), np.asarray(y)
    inside = shapely.contains_xy(aoi_geom, x, y)
    occ, x, y = occ[inside], x[inside], y[inside]
    steps.append(("inside AOI", len(occ)))

    rows, cols, ok = cell_index(stack.grid, x, y)
    occ, x, y, rows, cols = occ[ok], x[ok], y[ok], rows[ok], cols[ok]
    steps.append(("inside raster grid", len(occ)))
    if len(occ) == 0:
        raise ValueError("No occurrence records fall inside the AOI / raster grid.")

    vals = stack.sample_cells(rows, cols)
    finite = np.isfinite(vals).all(axis=1)
    occ, x, y, rows, cols, vals = (
        occ[finite],
        x[finite],
        y[finite],
        rows[finite],
        cols[finite],
        vals[finite],
    )
    steps.append(("complete predictor values", len(occ)))

    flat = rows * stack.grid.width + cols
    _, first = np.unique(flat, return_index=True)
    keep = np.sort(first)
    occ, x, y, rows, cols, vals = (
        occ.iloc[keep],
        x[keep],
        y[keep],
        rows[keep],
        cols[keep],
        vals[keep],
    )
    steps.append(("one record per raster cell", len(occ)))
    if len(occ) < 10:
        raise ValueError("Fewer than 10 usable presence cells remain.")

    pres = pd.DataFrame(vals, columns=stack.names)
    pres.insert(0, "source", "presence")
    pres.insert(1, "response", 1)
    pres.insert(2, "x", x)
    pres.insert(3, "y", y)
    pres.insert(4, "longitude", occ.longitude.to_numpy())
    pres.insert(5, "latitude", occ.latitude.to_numpy())
    pres["cell"] = rows * stack.grid.width + cols

    save_csv(
        pd.DataFrame(steps, columns=["step", "records_remaining"]),
        Path(out_dir) / "01a_presence_cleaning_log.csv",
        "presence cleaning log",
    )
    save_csv(
        pres,
        Path(out_dir) / "01_presence_records_cleaned.csv",
        "presence-only records used",
    )
    return pres


def sample_background(cfg, stack, aoi_geom, exclude_cells, rng, out_dir):
    """Uniform random background over valid, in-AOI cells (rejection sampling)."""
    g, need = stack.grid, int(cfg["N_BACKGROUND"])
    seen = set(int(c) for c in exclude_cells)
    kept = []
    n_kept, attempts = 0, 0
    for rnd in range(60):
        batch = max(2000, int((need - n_kept) * 3))
        r = rng.integers(0, g.height, batch)
        c = rng.integers(0, g.width, batch)
        flat = r * g.width + c
        _, uidx = np.unique(flat, return_index=True)
        r, c, flat = r[uidx], c[uidx], flat[uidx]
        fresh = np.array([f not in seen for f in flat])
        r, c, flat = r[fresh], c[fresh], flat[fresh]
        x, y = cell_centres(g, r, c)
        inside = shapely.contains_xy(aoi_geom, x, y)
        r, c, x, y, flat = r[inside], c[inside], x[inside], y[inside], flat[inside]
        attempts += batch
        if len(r) == 0:
            continue
        r, c, x, y, flat = (
            a[: int((need - n_kept) * 1.4) + 50] for a in (r, c, x, y, flat)
        )
        vals = stack.sample_cells(r, c)
        ok = np.isfinite(vals).all(axis=1)
        if ok.any():
            df = pd.DataFrame(vals[ok], columns=stack.names)
            df.insert(0, "source", "background")
            df.insert(1, "response", 0)
            df.insert(2, "x", x[ok])
            df.insert(3, "y", y[ok])
            df["cell"] = flat[ok]
            kept.append(df)
            n_kept += len(df)
            seen.update(int(f) for f in flat[ok])
        if n_kept >= need:
            break
    if not kept:
        raise ValueError("Could not draw any valid background points.")
    bg = pd.concat(kept, ignore_index=True).iloc[:need].copy()
    if len(bg) < need:
        log(
            f"WARNING: requested {need} background points but only {len(bg)} were valid."
        )
    if len(bg) < 100:
        raise ValueError("Insufficient valid background points.")
    tr = pyproj.Transformer.from_crs(g.crs.to_wkt(), 4326, always_xy=True)
    lon, lat = tr.transform(bg.x.to_numpy(), bg.y.to_numpy())
    bg.insert(4, "longitude", np.asarray(lon))
    bg.insert(5, "latitude", np.asarray(lat))
    save_csv(
        bg,
        Path(out_dir) / "02_background_pseudo_absence_points.csv",
        "background / pseudo-absence points used",
    )
    return bg


# =============================================================================
# 4. PREDICTOR FILTERING AND SPATIAL FOLDS
# =============================================================================
def select_predictors(df, names, threshold):
    x = df[names]
    usable = [n for n in names if np.nanstd(x[n].to_numpy()) > 0]
    dropped_const = sorted(set(names) - set(usable))
    corr = x[usable].corr(method="pearson").abs().fillna(0.0)
    selected, dropped_cor = [], []
    for n in usable:
        if not selected or corr.loc[n, selected].max() < threshold:
            selected.append(n)
        else:
            partner = corr.loc[n, selected].idxmax()
            dropped_cor.append((n, partner, float(corr.loc[n, partner])))
    return selected, dropped_const, dropped_cor


def make_spatial_folds(x, y, resp, crs, k, block_m, seed):
    tr = pyproj.Transformer.from_crs(crs.to_wkt(), 6933, always_xy=True)
    px, py = tr.transform(x, y)
    bx = np.floor(np.asarray(px) / block_m).astype("int64")
    by = np.floor(np.asarray(py) / block_m).astype("int64")
    pairs, inv = np.unique(np.c_[bx, by], axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    nb = len(pairs)
    if nb < k:
        raise ValueError(
            f"Only {nb} spatial blocks for {k} folds; reduce BLOCK_SIZE_M or K_FOLDS."
        )
    n_pres = np.bincount(inv, weights=(resp == 1), minlength=nb)
    n_bg = np.bincount(inv, weights=(resp == 0), minlength=nb)
    w = n_bg.sum() / max(n_pres.sum(), 1.0)
    load = n_pres * w + n_bg
    best, best_score = None, None
    for attempt in range(200):
        rng = np.random.default_rng(seed + attempt)
        order = np.lexsort((rng.random(nb), -load))
        fold_load = np.zeros(k)
        assign = np.zeros(nb, dtype=int)
        for b in order:
            cand = np.flatnonzero(fold_load == fold_load.min())
            f = int(rng.choice(cand))
            assign[b] = f
            fold_load[f] += load[b]
        fold_id = assign[inv]
        pres_per = np.array([(resp[fold_id == f] == 1).sum() for f in range(k)])
        bg_per = np.array([(resp[fold_id == f] == 0).sum() for f in range(k)])
        if pres_per.min() < 1 or bg_per.min() < 1:
            continue
        score = pres_per.max() - pres_per.min()
        if best is None or score < best_score:
            best, best_score = fold_id, score
        if pres_per.min() >= 2 and score <= max(2, 0.25 * pres_per.mean()):
            break
    if best is None:
        raise ValueError(
            "Could not build spatial folds with presences and background in each fold."
        )
    return best + 1  # folds numbered 1..k


# =============================================================================
# 5. MODEL WRAPPER
# =============================================================================
# def _add_quadratic(X):
#     return np.hstack([X, X**2])


class AddQuadratic(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype="float64")
        return np.hstack([X, X**2])


def _filter_kwargs(cls, kw):
    import inspect

    try:
        ok = set(inspect.signature(cls).parameters)
        return {k: v for k, v in kw.items() if k in ok}
    except (TypeError, ValueError):
        return kw


class SDMModel:
    """Uniform fit / predict interface. Class balance is equalised for every
    algorithm so that predicted probabilities are comparable in the ensemble."""

    def __init__(self, kind, hp=None, est_overrides=None, seed=0, n_jobs=1):
        self.kind = kind.upper()
        self.hp = {**DEFAULT_HP[self.kind], **(hp or {})}
        self.est_overrides = est_overrides or {}
        self.seed, self.n_jobs = seed, n_jobs
        self.estimator, self.lo, self.hi = None, None, None

    # -- estimator factory ------------------------------------------------------
    def build(self, X, n1, n0):
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            HistGradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler

        k, hp, seed = self.kind, self.hp, self.seed
        # if k == "GLM":
        #     est = make_pipeline(
        #         StandardScaler(),
        #         FunctionTransformer(_add_quadratic),
        #         StandardScaler(),
        #         LogisticRegression(C=hp["C"], max_iter=3000, class_weight="balanced"),
        #     )
        if k == "GLM":
            est = make_pipeline(
                StandardScaler(),
                AddQuadratic(),
                StandardScaler(),
                LogisticRegression(C=hp["C"], max_iter=3000, class_weight="balanced"),
            )
        elif k == "GAM":
            from pygam import LogisticGAM, l, s

            terms = None
            for i in range(X.shape[1]):
                nu = len(np.unique(X[:, i]))
                t = (
                    l(i)
                    if nu < 6
                    else s(i, n_splines=int(min(hp["n_splines"], nu - 1)))
                )
                terms = t if terms is None else terms + t
            est = LogisticGAM(terms, lam=hp["lam"], max_iter=hp["max_iter"])
        elif k == "SVM":
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.kernel_approximation import Nystroem
            from sklearn.svm import SVC, LinearSVC

            if hp["mode"] == "exact":
                est = make_pipeline(
                    StandardScaler(),
                    SVC(
                        kernel="rbf",
                        C=hp["C"],
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                )
            else:
                est = make_pipeline(
                    StandardScaler(),
                    Nystroem(
                        kernel="rbf",
                        gamma=hp["gamma"],
                        n_components=int(min(hp["n_components"], max(10, len(X)))),
                        random_state=seed,
                    ),
                    CalibratedClassifierCV(
                        LinearSVC(
                            C=hp["C"],
                            dual=False,
                            class_weight="balanced",
                            max_iter=5000,
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                )
        elif k == "MAXENT":
            kw = dict(
                feature_types=hp["feature_types"],
                beta_multiplier=hp["beta_multiplier"],
                n_hinge_features=hp["n_hinge_features"],
                transform="cloglog",
                n_cpus=self.n_jobs,
            )
            est = ela.MaxentModel(**_filter_kwargs(ela.MaxentModel, kw))
        elif k == "XGB":
            est = xgb.XGBClassifier(
                n_estimators=hp["n_estimators"],
                learning_rate=hp["learning_rate"],
                max_depth=hp["max_depth"],
                subsample=hp["subsample"],
                colsample_bytree=hp["colsample_bytree"],
                min_child_weight=hp["min_child_weight"],
                reg_lambda=hp["reg_lambda"],
                tree_method="hist",
                scale_pos_weight=n0 / max(n1, 1),
                n_jobs=self.n_jobs,
                random_state=seed,
                eval_metric="logloss",
                verbosity=0,
            )
        elif k == "RF":
            est = RandomForestClassifier(
                n_estimators=hp["n_estimators"],
                max_features=hp["max_features"],
                min_samples_leaf=hp["min_samples_leaf"],
                class_weight="balanced_subsample",
                n_jobs=self.n_jobs,
                random_state=seed,
            )
        elif k == "BRT":
            if hp["engine"] == "classic":
                est = GradientBoostingClassifier(
                    n_estimators=hp["n_estimators"],
                    learning_rate=hp["learning_rate"],
                    max_depth=hp["max_depth"],
                    subsample=hp["subsample"],
                    min_samples_leaf=hp["min_samples_leaf"],
                    random_state=seed,
                )
            else:
                est = HistGradientBoostingClassifier(
                    max_iter=hp["n_estimators"],
                    learning_rate=hp["learning_rate"],
                    max_depth=hp["max_depth"],
                    min_samples_leaf=hp["min_samples_leaf"],
                    l2_regularization=1.0,
                    random_state=seed,
                )
        else:
            raise ValueError(f"Unknown model: {k}")
        if self.est_overrides:
            est.set_params(**self.est_overrides)
        return est

    # -- fit / predict --------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y).astype(int)
        self.lo, self.hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        n, n1 = len(y), int(y.sum())
        n0 = n - n1
        w = np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))
        est = self.build(X, n1, n0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.kind == "GAM":
                est.fit(X, y, weights=w)
            elif self.kind == "BRT":
                est.fit(X, y, sample_weight=w)
            else:
                est.fit(X, y)
        self.estimator = est
        return self

    def _predict_batch(self, xb):
        k, est = self.kind, self.estimator
        if k == "GAM":
            return est.predict_mu(xb)
        if k == "MAXENT":
            return est.predict(xb)
        return est.predict_proba(xb)[:, 1]

    def predict(self, X, batch=250_000):
        X = np.asarray(X)
        out = np.empty(len(X), dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(0, len(X), batch):
                xb = np.clip(
                    X[i : i + batch].astype("float64"), self.lo, self.hi
                )  # clamp
                out[i : i + batch] = np.clip(self._predict_batch(xb), 0.0, 1.0)
        return out

    def export(self, features):
        return dict(
            name=self.kind,
            features=list(features),
            lo=self.lo,
            hi=self.hi,
            hyperparameters=self.hp,
            estimator_overrides=self.est_overrides,
            estimator=self.estimator,
            note="Clip inputs to [lo, hi] before predicting; GAM->predict_mu, "
            "MAXENT->predict, others->predict_proba[:,1].",
        )


def _cv_task(fold, name, Xtr, ytr, Xte, hp, est_over, seed):
    t0 = time.time()
    try:
        m = SDMModel(name, hp=hp, est_overrides=est_over, seed=seed, n_jobs=1).fit(
            Xtr, ytr
        )
        return fold, name, m.predict(Xte), None, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return fold, name, None, f"{type(e).__name__}: {e}", time.time() - t0


def _final_task(name, X, y, hp, est_over, seed):
    t0 = time.time()
    try:
        m = SDMModel(name, hp=hp, est_overrides=est_over, seed=seed, n_jobs=1).fit(X, y)
        return name, m, None, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return name, None, f"{type(e).__name__}: {e}", time.time() - t0


# =============================================================================
# 6. METRICS AND ENSEMBLE WEIGHTS
# =============================================================================
def boyce_index(p_background, p_presence, n_bins=100, window=0.1):
    lo = min(np.min(p_background), np.min(p_presence))
    hi = max(np.max(p_background), np.max(p_presence))
    if hi <= lo or len(p_presence) < 3:
        return np.nan
    w = window * (hi - lo)
    centres = np.linspace(lo + w / 2, hi - w / 2, n_bins)
    pe = np.full(n_bins, np.nan)
    for i, c in enumerate(centres):
        a, b = c - w / 2, c + w / 2
        e = np.mean((p_background >= a) & (p_background <= b))
        if e > 0:
            pe[i] = np.mean((p_presence >= a) & (p_presence <= b)) / e
    ok = np.isfinite(pe)
    if ok.sum() < 3 or np.ptp(pe[ok]) == 0:
        return np.nan
    return float(scipy_stats.spearmanr(centres[ok], pe[ok]).correlation)


def evaluate(y, p):
    from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    res = dict(
        n_test_presence=int(y.sum()),
        n_test_background=int((1 - y).sum()),
        auc=np.nan,
        tss=np.nan,
        threshold_max_tss=np.nan,
        sensitivity=np.nan,
        specificity=np.nan,
        brier=np.nan,
        boyce=np.nan,
    )
    if len(np.unique(y)) < 2:
        return res
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    i = int(np.argmax(j))
    t = float(thr[i]) if np.isfinite(thr[i]) else 0.5
    res.update(
        auc=float(roc_auc_score(y, p)),
        tss=float(j[i]),
        threshold_max_tss=t,
        sensitivity=float(tpr[i]),
        specificity=float(1 - fpr[i]),
        brier=float(brier_score_loss(y, np.clip(p, 0, 1))),
        boyce=boyce_index(p[y == 0], p[y == 1]),
    )
    return res


def compute_weights(fold_metrics, models, cfg):
    method, power = cfg["WEIGHT_METHOD"].lower(), float(cfg["WEIGHT_POWER"])
    rows = []
    for m in models:
        d = fold_metrics[(fold_metrics.model == m) & (fold_metrics.fold != "pooled")]
        pooled = fold_metrics[
            (fold_metrics.model == m) & (fold_metrics.fold == "pooled")
        ]
        mean_auc = d.auc.astype(float).mean()
        mean_tss = d.tss.astype(float).mean()
        mean_boyce = d.boyce.astype(float).mean()
        if method == "auc":
            raw = max(mean_auc - 0.5, 0.0) ** power
        elif method == "tss":
            raw = max(mean_tss, 0.0) ** power
        elif method == "boyce":
            raw = max(mean_boyce, 0.0) ** power
        else:
            raw = 1.0
        include = bool(
            np.isfinite(mean_auc) and mean_auc >= cfg["MIN_MEAN_AUC"] and raw > 0
        )
        rows.append(
            dict(
                model=m,
                mean_fold_auc=mean_auc,
                mean_fold_tss=mean_tss,
                mean_fold_boyce=mean_boyce,
                pooled_auc=float(pooled.auc.iloc[0]) if len(pooled) else np.nan,
                raw_weight=raw if include else 0.0,
                included_in_ensemble=include,
            )
        )
    w = pd.DataFrame(rows)
    tot = w.raw_weight.sum()
    if tot <= 0:
        log(
            "WARNING: no model passed the weight criteria; using equal weights for all fitted models."
        )
        w["raw_weight"] = 1.0
        w["included_in_ensemble"] = True
        tot = w.raw_weight.sum()
    w["weight"] = w.raw_weight / tot
    w["weighting_method"] = method
    return w


# =============================================================================
# 7. OPTIONAL QUICK TUNING
# =============================================================================
def quick_tune(X, y, fold_ids, cfg, n_jobs, out_dir):
    from sklearn.model_selection import RandomizedSearchCV

    splits = [
        (np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f))
        for f in np.unique(fold_ids)
    ]
    n1 = int(y.sum())
    overrides, rows = {}, []
    for name in [m for m in cfg["MODELS"] if m in TUNE_GRIDS]:
        hp = {**DEFAULT_HP[name], **cfg["HP"].get(name, {})}
        if name == "BRT" and hp["engine"] == "classic":
            continue
        base = SDMModel(name, hp=hp, seed=cfg["SEED"], n_jobs=1).build(
            X, n1, len(y) - n1
        )
        t0 = time.time()
        rs = RandomizedSearchCV(
            base,
            TUNE_GRIDS[name],
            n_iter=cfg["TUNE_ITER"],
            scoring="roc_auc",
            cv=splits,
            n_jobs=n_jobs,
            random_state=cfg["SEED"],
            refit=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "BRT":
                w = np.where(
                    y == 1, len(y) / (2.0 * n1), len(y) / (2.0 * (len(y) - n1))
                )
                rs.fit(X, y, sample_weight=w)
            else:
                rs.fit(X, y)
        overrides[name] = rs.best_params_
        res = pd.DataFrame(rs.cv_results_)
        res.insert(0, "model", name)
        rows.append(res)
        log(
            f"tuned {name}: best AUC {rs.best_score_:.3f} {rs.best_params_} ({time.time() - t0:,.0f}s)"
        )
        save_csv(
            pd.concat(rows, ignore_index=True),
            Path(out_dir) / "03c_tuning_results.csv",
            "tuning results",
        )
    return overrides


# =============================================================================
# 8. RASTER PREDICTION (one pass, ensemble + per-model rasters)
# =============================================================================
def predict_rasters(
    stack, aoi_geom, fitted, weights_df, threshold, features, cfg, out_dir
):
    g = stack.grid
    nodata = float(cfg["NODATA"])
    w_map = dict(zip(weights_df.model, weights_df.weight))
    ens_models = [m for m in fitted if w_map.get(m, 0) > 0]
    write_models = list(fitted) if cfg["WRITE_MODEL_RASTERS"] else []
    need = sorted(set(ens_models) | set(write_models))
    wvec = np.array([w_map.get(m, 0.0) for m in ens_models], dtype="float64")
    wvec = wvec / wvec.sum()

    rdir = Path(out_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    base = dict(
        driver="GTiff",
        height=g.height,
        width=g.width,
        count=1,
        crs=g.crs,
        transform=g.transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=3,
        BIGTIFF="IF_SAFER",
    )
    paths = {
        "ENSEMBLE_SUITABILITY": rdir / f"suitability_ENSEMBLE_{cfg['RUN_LABEL']}.tif",
        "ENSEMBLE_UNCERTAINTY_SD": rdir
        / f"uncertainty_sd_ENSEMBLE_{cfg['RUN_LABEL']}.tif",
        "ENSEMBLE_BINARY": rdir / f"binary_ENSEMBLE_maxTSS_{cfg['RUN_LABEL']}.tif",
    }
    for m in write_models:
        paths[m] = rdir / f"suitability_{m}_{cfg['RUN_LABEL']}.tif"
    dsts = {}
    for k, p in paths.items():
        if k == "ENSEMBLE_BINARY":
            dsts[k] = rasterio.open(
                p, "w", dtype="uint8", nodata=255, **{**base, "predictor": 2}
            )
        else:
            dsts[k] = rasterio.open(p, "w", dtype="float32", nodata=nodata, **base)

    rows_per = max(256, (int(cfg["CHUNK_CELLS"]) // g.width) // 256 * 256)
    wins = [
        Window(0, r0, g.width, min(rows_per, g.height - r0))
        for r0 in range(0, g.height, rows_per)
    ]
    log(
        f"raster grid {g.width} x {g.height} ({g.width * g.height:,} cells), "
        f"{len(wins)} blocks, models predicted: {need}"
    )

    def load(win):
        b = rasterio.windows.bounds(win, g.transform)
        geom = shapely.clip_by_rect(aoi_geom, *b)
        if geom.is_empty:
            return win, None, None
        aoi = rasterio.features.geometry_mask(
            [geom],
            out_shape=(int(win.height), int(win.width)),
            transform=rasterio.windows.transform(win, g.transform),
            invert=True,
        )
        if not aoi.any():
            return win, None, None
        return win, aoi, stack.read_window(win, features)

    stats = {
        k: dict(n=0, s=0.0, mn=np.inf, mx=-np.inf)
        for k in dsts
        if k != "ENSEMBLE_BINARY"
    }

    def upd(k, v):
        d = stats[k]
        d["n"] += v.size
        d["s"] += float(v.sum(dtype="float64"))
        d["mn"] = min(d["mn"], float(v.min()))
        d["mx"] = max(d["mx"], float(v.max()))

    loader = ThreadPoolExecutor(max_workers=1)
    mpool = ThreadPoolExecutor(max_workers=max(1, len(need)))
    t0 = time.time()
    try:
        fut = loader.submit(load, wins[0])
        for i, win in enumerate(wins):
            win, aoi, arr = fut.result()
            if i + 1 < len(wins):
                fut = loader.submit(load, wins[i + 1])
            shape = (int(win.height), int(win.width))
            outs = {
                k: np.full(shape, nodata, dtype="float32")
                for k in dsts
                if k != "ENSEMBLE_BINARY"
            }
            binary = np.full(shape, 255, dtype="uint8")
            if aoi is not None:
                valid = aoi & np.isfinite(arr).all(axis=0)
                if valid.any():
                    X = np.ascontiguousarray(arr[:, valid].T)
                    futs = {m: mpool.submit(fitted[m].predict, X) for m in need}
                    P = {m: f.result() for m, f in futs.items()}
                    if ens_models:
                        M = np.column_stack([P[m] for m in ens_models]).astype(
                            "float64"
                        )
                        mean = M @ wvec
                        sd = np.sqrt(((M - mean[:, None]) ** 2) @ wvec)
                        outs["ENSEMBLE_SUITABILITY"][valid] = mean
                        outs["ENSEMBLE_UNCERTAINTY_SD"][valid] = sd
                        binary[valid] = (mean >= threshold).astype("uint8")
                        upd("ENSEMBLE_SUITABILITY", mean)
                        upd("ENSEMBLE_UNCERTAINTY_SD", sd)
                    for m in write_models:
                        outs[m][valid] = P[m]
                        upd(m, P[m])
            for k, a in outs.items():
                dsts[k].write(a, 1, window=win)
            dsts["ENSEMBLE_BINARY"].write(binary, 1, window=win)
            if (i + 1) % max(1, len(wins) // 10) == 0 or i + 1 == len(wins):
                log(
                    f"  prediction progress {i + 1}/{len(wins)} blocks ({time.time() - t0:,.0f}s)"
                )
    finally:
        loader.shutdown(wait=False)
        mpool.shutdown(wait=False)
        for d in dsts.values():
            d.close()

    rows = []
    for k, d in stats.items():
        rows.append(
            dict(
                raster=k,
                file=str(paths[k]),
                valid_cells=d["n"],
                min=d["mn"] if d["n"] else np.nan,
                mean=d["s"] / d["n"] if d["n"] else np.nan,
                max=d["mx"] if d["n"] else np.nan,
            )
        )
    rows.append(
        dict(
            raster="ENSEMBLE_BINARY",
            file=str(paths["ENSEMBLE_BINARY"]),
            valid_cells=stats["ENSEMBLE_SUITABILITY"]["n"],
            min=0,
            mean=np.nan,
            max=1,
        )
    )
    summary = pd.DataFrame(rows)
    save_csv(
        summary,
        rdir.parent / "csv" / "09_output_raster_summary.csv",
        "output raster summary",
    )
    return paths, summary


# =============================================================================
# 9. REPRODUCIBILITY REPORT
# =============================================================================
def fingerprint(path, full_max_mb):
    p = Path(path)
    st = p.stat()
    size = st.st_size
    full = size <= full_max_mb * 1024 * 1024
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if full:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        else:
            h.update(f.read(1 << 24))
            f.seek(max(0, size - (1 << 24)))
            h.update(f.read(1 << 24))
            h.update(str(size).encode())
    return dict(
        path=str(p),
        size_bytes=size,
        modified_utc=dt.datetime.fromtimestamp(
            st.st_mtime, dt.timezone.utc
        ).isoformat(),
        sha256=h.hexdigest(),
        hash_scope="full" if full else "partial (first 16MB + last 16MB + size)",
    )


def _git(args, cwd):
    try:
        r = subprocess.run(
            ["git", "-C", str(cwd)] + args, capture_output=True, text=True, timeout=15
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _norm_url(u):
    if not u:
        return ""
    u = u.strip()
    m = re.match(r"git@([^:]+):(.+)", u)
    if m:
        u = f"https://{m.group(1)}/{m.group(2)}"
    return re.sub(r"(\.git)?/?$", "", u).lower()


def _http_check(url, timeout=12):
    try:
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "sdm-repro-check"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return dict(reachable=True, status=r.status, final_url=r.geturl())
    except Exception as e1:  # noqa: BLE001
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sdm-repro-check"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return dict(reachable=True, status=r.status, final_url=r.geturl())
        except Exception as e2:  # noqa: BLE001
            return dict(
                reachable=False, error=f"{type(e2).__name__}: {e2}", first_error=str(e1)
            )


def snapshot_source(out_dir):
    """Copy the executing code next to the results (script file or notebook cell)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if "__file__" in globals() and Path(__file__).exists():
            dst = out / Path(__file__).name
            shutil.copy2(__file__, dst)
            return str(Path(__file__).resolve()), str(dst)
    except Exception:  # noqa: BLE001
        pass
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        cells = [
            c
            for c in ip.user_ns.get("In", [])
            if "SDM_ENSEMBLE_SCRIPT_MARKER" in c and "def main" in c
        ]
        if cells:
            dst = out / "sdm_ensemble_7models_notebook_cell.py"
            dst.write_text(max(cells, key=len), encoding="utf-8")
            return "jupyter cell", str(dst)
    except Exception:  # noqa: BLE001
        pass
    return None, None


def write_reproducibility_report(cfg, ctx, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warn = []
    script_path, snapshot = snapshot_source(out)
    script_fp = fingerprint(snapshot, cfg["HASH_FULL_MAX_MB"]) if snapshot else None
    if snapshot is None:
        warn.append(
            "Could not snapshot the executing code (no __file__ and no notebook cell found)."
        )

    # ---- git / repository ---------------------------------------------------
    cwd = (
        Path(script_path).parent
        if script_path and script_path != "jupyter cell"
        else Path.cwd()
    )
    root = _git(["rev-parse", "--show-toplevel"], cwd)
    git = dict(is_repository=bool(root), search_dir=str(cwd))
    if root:
        status = _git(["status", "--porcelain"], root) or ""
        git.update(
            root=root,
            commit=_git(["rev-parse", "HEAD"], root),
            branch=_git(["rev-parse", "--abbrev-ref", "HEAD"], root),
            describe=_git(["describe", "--tags", "--always", "--dirty"], root),
            remote_origin=_git(["config", "--get", "remote.origin.url"], root),
            working_tree_clean=(status == ""),
            uncommitted_files=status.splitlines()[:50],
            commit_on_remote_branch=bool(
                _git(["branch", "-r", "--contains", "HEAD"], root)
            ),
        )
        if not git["working_tree_clean"]:
            warn.append(
                "Git working tree has uncommitted changes; the commit hash does not fully describe the code."
            )
        if not git["commit_on_remote_branch"]:
            warn.append(
                "The current commit is not contained in any remote branch (not pushed)."
            )
    else:
        warn.append("Code is not inside a git repository.")

    repo_v = dict(configured_url=cfg["CODE_REPO_URL"] or None)
    if cfg["CODE_REPO_URL"]:
        repo_v["matches_git_remote"] = _norm_url(cfg["CODE_REPO_URL"]) == _norm_url(
            git.get("remote_origin")
        )
        if not repo_v["matches_git_remote"]:
            warn.append("CODE_REPO_URL does not match the git remote 'origin'.")
        if cfg["VERIFY_ONLINE"]:
            repo_v["online_check"] = _http_check(cfg["CODE_REPO_URL"])
            if not repo_v["online_check"].get("reachable"):
                warn.append(
                    "Repository URL could not be reached (offline or wrong URL)."
                )
    else:
        warn.append("CODE_REPO_URL not set.")

    arch = dict(doi=cfg["CODE_ARCHIVE_DOI"] or None)
    if cfg["CODE_ARCHIVE_DOI"]:
        doi = cfg["CODE_ARCHIVE_DOI"].replace("https://doi.org/", "").strip()
        arch["doi_format_valid"] = bool(re.match(r"^10\.\d{4,9}/\S+$", doi))
        if cfg["VERIFY_ONLINE"]:
            arch["online_check"] = _http_check(f"https://doi.org/{doi}")
            if not arch["online_check"].get("reachable"):
                warn.append("Archive DOI could not be resolved (offline or wrong DOI).")
    else:
        warn.append("CODE_ARCHIVE_DOI not set.")
    if cfg["CODE_ARCHIVE_FILE"]:
        fp = fingerprint(cfg["CODE_ARCHIVE_FILE"], 1 << 20)
        arch["archive_file"] = fp
        if cfg["CODE_ARCHIVE_SHA256"]:
            arch["checksum_matches_expected"] = (
                fp["sha256"].lower() == cfg["CODE_ARCHIVE_SHA256"].lower()
            )
            if not arch["checksum_matches_expected"]:
                warn.append("Archive checksum does not match CODE_ARCHIVE_SHA256.")

    # ---- environment ----------------------------------------------------------
    pkgs = {}
    for p in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "rasterio",
        "geopandas",
        "shapely",
        "pyproj",
        "xgboost",
        "pygam",
        "elapid",
    ]:
        try:
            pkgs[p] = md.version(p)
        except Exception:  # noqa: BLE001
            pkgs[p] = None
    try:
        gdal = rasterio.__gdal_version__
    except Exception:  # noqa: BLE001
        gdal = None
    try:
        freeze = sorted(
            f"{d.metadata['Name']}=={d.version}" for d in md.distributions()
        )
        (out / "requirements_snapshot.txt").write_text(
            "\n".join(freeze), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001
        pass

    # ---- inputs ---------------------------------------------------------------
    shp = Path(cfg["AOI_FILE"])
    aoi_files = [p for p in shp.parent.glob(shp.stem + ".*")]
    inputs = dict(
        occurrence_csv=fingerprint(cfg["OCCURRENCE_CSV"], cfg["HASH_FULL_MAX_MB"]),
        aoi_files=[fingerprint(p, cfg["HASH_FULL_MAX_MB"]) for p in aoi_files],
        predictor_rasters=[
            fingerprint(p, cfg["HASH_FULL_MAX_MB"]) for p in ctx["raster_files"]
        ],
    )

    outputs = []
    for p in sorted(Path(cfg["OUTPUT_DIR"]).rglob("*")):
        if (
            p.is_file()
            and p.suffix.lower() in {".csv", ".tif", ".joblib", ".json", ".txt"}
            and p.name
            not in {"reproducibility_report.json", "reproducibility_report.txt"}
        ):
            fp = fingerprint(p, cfg["HASH_FULL_MAX_MB"])
            outputs.append(
                dict(
                    file=str(p.relative_to(cfg["OUTPUT_DIR"])),
                    size_bytes=fp["size_bytes"],
                    sha256=fp["sha256"],
                    hash_scope=fp["hash_scope"],
                )
            )

    report = dict(
        script_version=SCRIPT_VERSION,
        run_label=cfg["RUN_LABEL"],
        started_utc=ctx["started_utc"],
        finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        total_runtime_seconds=round(time.time() - ctx["t0"], 1),
        python=sys.version,
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        gdal_version=gdal,
        packages=pkgs,
        random_seed=cfg["SEED"],
        run_configuration=cfg,
        effective_hyperparameters=ctx["effective_hp"],
        selected_predictors=ctx["features"],
        dropped_constant=ctx["dropped_const"],
        dropped_correlated=ctx["dropped_cor"],
        n_presence=ctx["n_pres"],
        n_background=ctx["n_bg"],
        ensemble_weights=ctx["weights"],
        ensemble_threshold_max_tss=ctx["threshold"],
        source_code=dict(executed_from=script_path, snapshot=script_fp),
        git=git,
        repository_verification=repo_v,
        archive_verification=arch,
        inputs=inputs,
        outputs=outputs,
        warnings=warn,
        reproducibility_ready=(len(warn) == 0),
    )
    save_json(report, out / "reproducibility_report.json")

    lines = (
        [
            f"SDM ENSEMBLE REPRODUCIBILITY REPORT  (script v{SCRIPT_VERSION}, run '{cfg['RUN_LABEL']}')",
            "=" * 78,
            f"Started  : {report['started_utc']}",
            f"Finished : {report['finished_utc']}",
            f"Python   : {sys.version.split()[0]} | {platform.platform()} | GDAL {gdal}",
            f"Seed     : {cfg['SEED']}",
            f"Code     : {script_path}  (sha256 {script_fp['sha256'][:16] + '...' if script_fp else 'n/a'})",
            f"Git      : commit {git.get('commit')} branch {git.get('branch')} "
            f"clean={git.get('working_tree_clean')} remote={git.get('remote_origin')}",
            f"Repo URL : {cfg['CODE_REPO_URL'] or 'not set'}  -> {repo_v.get('online_check', 'not checked')}",
            f"Archive  : DOI {cfg['CODE_ARCHIVE_DOI'] or 'not set'}  -> {arch.get('online_check', 'not checked')}",
            "",
            "Packages:",
        ]
        + [f"  {k}: {v}" for k, v in pkgs.items()]
        + [
            "",
            f"Presences: {ctx['n_pres']}   Background: {ctx['n_bg']}",
            f"Predictors used ({len(ctx['features'])}): {', '.join(ctx['features'])}",
            f"Ensemble max-TSS threshold: {ctx['threshold']:.4f}",
            "Ensemble weights:",
        ]
        + [
            f"  {w['model']:<7} {w['weight']:.4f}  (mean fold AUC {w['mean_fold_auc']:.3f})"
            for w in ctx["weights"]
        ]
        + ["", "Warnings:"]
        + ([f"  - {w}" for w in warn] or ["  none"])
        + [
            "",
            "Full details (input/output checksums, configuration): reproducibility_report.json",
        ]
    )
    (out / "reproducibility_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"saved reproducibility report -> {out / 'reproducibility_report.json'}")
    for w in warn:
        log(f"REPRODUCIBILITY WARNING: {w}")
    return report


# =============================================================================
# 10. MAIN
# =============================================================================
def main(overrides=None):
    cfg = {**CONFIG, **(overrides or {})}
    cfg["HP"] = {k.upper(): v for k, v in cfg.get("HP", {}).items()}
    cfg["MODELS"] = [m.upper() for m in cfg["MODELS"]]
    TIMINGS.clear()

    need_mod = dict(
        rasterio=rasterio,
        geopandas=gpd,
        shapely=shapely,
        pyproj=pyproj,
        joblib=joblib,
        sklearn=sklearn,
        scipy=scipy_stats,
    )
    if "XGB" in cfg["MODELS"]:
        need_mod["xgboost"] = xgb
    if "GAM" in cfg["MODELS"]:
        need_mod["pygam"] = pygam
    if "MAXENT" in cfg["MODELS"]:
        need_mod["elapid"] = ela
    missing = [k for k, v in need_mod.items() if v is None]
    if missing:
        raise ImportError(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall with: pip install "
            + " ".join(
                {"sklearn": "scikit-learn", "scipy": "scipy"}.get(m, m) for m in missing
            )
        )
    if int(shapely.__version__.split(".")[0]) < 2:
        raise ImportError("shapely >= 2.0 is required.")

    for key in ("OCCURRENCE_CSV", "AOI_FILE"):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f"{key} not found: {cfg[key]}")
    if not Path(cfg["RASTER_DIR"]).is_dir():
        raise FileNotFoundError(f"RASTER_DIR not found: {cfg['RASTER_DIR']}")

    out = Path(cfg["OUTPUT_DIR"])
    csv_dir, rst_dir, mdl_dir, rep_dir = (
        out / "csv",
        out / "rasters",
        out / "models",
        out / "reproducibility",
    )
    for d in (csv_dir, rst_dir, mdl_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    t_run, started = time.time(), dt.datetime.now(dt.timezone.utc).isoformat()
    n_cpu = os.cpu_count() or 1
    n_jobs = n_cpu if int(cfg["N_JOBS"]) in (-1, 0) else int(cfg["N_JOBS"])
    rng = np.random.default_rng(cfg["SEED"])
    log(
        f"SDM ensemble v{SCRIPT_VERSION} | run '{cfg['RUN_LABEL']}' | {n_jobs} workers | models: {cfg['MODELS']}"
    )
    save_json(
        dict(script_version=SCRIPT_VERSION, started_utc=started, config=cfg),
        rep_dir / "run_configuration_initial.json",
    )

    # ------------------------------------------------------------------ AOI + grid
    with stage("load AOI and predictor stack", out):
        files = sorted(
            p
            for p in Path(cfg["RASTER_DIR"]).rglob("*")
            if p.suffix.lower() in (".tif", ".tiff")
        )
        if not files:
            raise FileNotFoundError(f"No GeoTIFF predictors in {cfg['RASTER_DIR']}")
        log(f"{len(files)} predictor rasters found")
        used = set()
        names = [clean_name(p.stem, used) for p in files]
        ref = rasterio.open(files[0])
        if ref.crs is None:
            raise ValueError("Reference raster has no CRS.")
        aoi = gpd.read_file(cfg["AOI_FILE"])
        if aoi.crs is None:
            raise ValueError("AOI shapefile has no CRS.")
        aoi = aoi.to_crs(ref.crs.to_wkt())
        aoi_geom = shapely.make_valid(
            shapely.union_all(shapely.make_valid(aoi.geometry.values))
        )
        grid = build_grid(ref, aoi_geom)
        ref.close()
        log(f"prediction grid: {grid.width} x {grid.height} cells, CRS {grid.crs}")
        resamp = getattr(Resampling, cfg["RESAMPLING"])
        stack = PredictorStack(
            files, names, grid, resamp, n_threads=min(16, max(4, n_cpu))
        )
        modes = pd.Series([L.mode for L in stack.layers]).value_counts().to_dict()
        log(f"predictor alignment modes (direct=no resampling needed): {modes}")
        save_csv(
            pd.DataFrame(
                dict(
                    predictor=names,
                    raster_file=[str(f) for f in files],
                    alignment=[L.mode for L in stack.layers],
                )
            ),
            csv_dir / "00_predictor_inventory.csv",
            "predictor inventory",
        )

    try:
        # -------------------------------------------------------------- data
        with stage("presence records", out):
            pres = load_presences(cfg, stack, aoi_geom, csv_dir)
            log(f"presence cells used: {len(pres):,}")
        with stage("background sampling", out):
            bg = sample_background(
                cfg, stack, aoi_geom, pres["cell"].to_numpy(), rng, csv_dir
            )

        data = pd.concat([pres, bg], ignore_index=True)
        data.insert(0, "row_id", np.arange(len(data)))
        feats, dropped_const, dropped_cor = select_predictors(
            data, names, cfg["COR_THRESHOLD"]
        )
        if not feats:
            raise ValueError("No usable predictors remain after filtering.")
        log(
            f"predictors kept {len(feats)}/{len(names)}; constant dropped: {dropped_const}; "
            f"correlated dropped: {[d[0] for d in dropped_cor]}"
        )
        save_csv(
            pd.DataFrame(dict(predictor=feats)),
            csv_dir / "03a_selected_predictors.csv",
            "selected predictors",
        )
        dc = pd.DataFrame(
            dropped_cor,
            columns=["dropped_predictor", "correlated_with", "abs_correlation"],
        )
        dc = pd.concat(
            [
                dc,
                pd.DataFrame(
                    dict(
                        dropped_predictor=dropped_const,
                        correlated_with="(constant)",
                        abs_correlation=np.nan,
                    )
                ),
            ]
        )
        save_csv(dc, csv_dir / "03b_dropped_predictors.csv", "dropped predictors")

        # -------------------------------------------------------------- folds
        with stage("spatial block folds", out):
            data["fold"] = make_spatial_folds(
                data.x.to_numpy(),
                data.y.to_numpy(),
                data.response.to_numpy(),
                grid.crs,
                cfg["K_FOLDS"],
                cfg["BLOCK_SIZE_M"],
                cfg["SEED"],
            )
            first = ["row_id", "source", "response", "fold", "x", "y"]
            save_csv(
                data[first + [c for c in data.columns if c not in first]],
                csv_dir / "04_model_dataset_with_folds.csv",
                "model dataset (all training/testing records + fold)",
            )
            save_csv(
                data[["row_id", "source", "response", "x", "y", "fold"]],
                csv_dir / "04a_cv_fold_assignments.csv",
                "cross-validation fold assignments",
            )
            fsum = (
                data.groupby("fold")
                .agg(n=("row_id", "size"), presences=("response", "sum"))
                .reset_index()
            )
            fsum["background"] = fsum.n - fsum.presences
            save_csv(fsum, csv_dir / "04b_cv_fold_summary.csv", "fold summary")
            member = pd.concat(
                [
                    data[["row_id", "response", "x", "y"]].assign(
                        fold=f, role=np.where(data.fold == f, "test", "train")
                    )
                    for f in range(1, cfg["K_FOLDS"] + 1)
                ]
            )
            save_csv(
                member[["fold", "row_id", "role", "response", "x", "y"]],
                csv_dir / "04c_train_test_membership_long.csv",
                "train/test membership per fold",
            )
            if cfg["SAVE_PER_FOLD_CSVS"]:
                fdir = csv_dir / "folds"
                for f in range(1, cfg["K_FOLDS"] + 1):
                    cols = first + feats
                    save_csv(
                        data.loc[data.fold != f, cols],
                        fdir / f"fold_{f}_train.csv",
                        f"fold {f} training data",
                    )
                    save_csv(
                        data.loc[data.fold == f, cols],
                        fdir / f"fold_{f}_test.csv",
                        f"fold {f} test data",
                    )

        X = data[feats].to_numpy(dtype="float64")
        y = data.response.to_numpy(dtype=int)
        fold_ids = data.fold.to_numpy()
        n_pres, n_bg = int(y.sum()), int((1 - y).sum())
        log(
            f"modelling dataset: {len(data):,} rows ({n_pres} presences, {n_bg} background)"
        )

        # -------------------------------------------------------------- tuning
        est_over = {}
        if cfg["TUNE"]:
            with stage("quick hyperparameter tuning (spatial CV)", out):
                est_over = quick_tune(X, y, fold_ids, cfg, n_jobs, csv_dir)

        def hp_of(m):
            return {**DEFAULT_HP[m], **cfg["HP"].get(m, {})}

        effective_hp = {
            m: dict(hp=hp_of(m), estimator_overrides=est_over.get(m, {}))
            for m in cfg["MODELS"]
        }
        save_json(effective_hp, csv_dir / "05_effective_hyperparameters.json")

        # -------------------------------------------------------------- CV
        with stage("spatial cross-validation (all models x folds in parallel)", out):
            tasks = []
            for f in range(1, cfg["K_FOLDS"] + 1):
                tr, te = fold_ids != f, fold_ids == f
                for m in cfg["MODELS"]:
                    tasks.append(
                        (
                            f,
                            m,
                            X[tr],
                            y[tr],
                            X[te],
                            hp_of(m),
                            est_over.get(m, {}),
                            cfg["SEED"] + 1000 * f,
                        )
                    )
            oof = {
                m: np.full(len(data), np.nan, dtype="float32") for m in cfg["MODELS"]
            }
            fit_log = []
            held = data[["row_id", "fold", "response", "x", "y"]].copy()

            def handle(res):
                f, m, p, err, secs = res
                if err:
                    log(f"  [fold {f}] {m} FAILED: {err}")
                else:
                    oof[m][fold_ids == f] = p
                fit_log.append(dict(fold=f, model=m, seconds=round(secs, 2), error=err))
                part = held.copy()
                for mm in cfg["MODELS"]:
                    part[mm] = oof[mm]
                save_csv(
                    part,
                    csv_dir / "06_heldout_predictions_partial.csv",
                    "held-out predictions (partial)",
                )

            gen = (
                joblib.Parallel(
                    n_jobs=min(n_jobs, len(tasks)), return_as="generator_unordered"
                )
                if _joblib_has_generator()
                else None
            )
            if gen is not None:
                for res in gen(joblib.delayed(_cv_task)(*t) for t in tasks):
                    handle(res)
            else:
                for res in joblib.Parallel(n_jobs=min(n_jobs, len(tasks)))(
                    joblib.delayed(_cv_task)(*t) for t in tasks
                ):
                    handle(res)
            save_csv(
                pd.DataFrame(fit_log), csv_dir / "06a_cv_fit_log.csv", "CV fit log"
            )
            ok_models = [m for m in cfg["MODELS"] if np.isfinite(oof[m]).all()]
            failed = [m for m in cfg["MODELS"] if m not in ok_models]
            if failed:
                log(f"WARNING: models with failed folds are excluded: {failed}")
            if not ok_models:
                raise RuntimeError("All models failed in cross-validation.")

        # -------------------------------------------------------------- metrics + weights
        with stage("metrics and ensemble weights", out):
            mrows = []
            for m in ok_models:
                for f in range(1, cfg["K_FOLDS"] + 1):
                    sel = fold_ids == f
                    mrows.append(dict(model=m, fold=f, **evaluate(y[sel], oof[m][sel])))
                mrows.append(dict(model=m, fold="pooled", **evaluate(y, oof[m])))
            fm = pd.DataFrame(mrows)
            weights = compute_weights(fm, ok_models, cfg)
            save_csv(weights, csv_dir / "08_ensemble_weights.csv", "ensemble weights")

            wm = dict(zip(weights.model, weights.weight))
            M = np.column_stack([oof[m] for m in ok_models]).astype("float64")
            ens_oof = M @ np.array([wm[m] for m in ok_models])
            oof["ENSEMBLE"] = ens_oof.astype("float32")
            for f in range(1, cfg["K_FOLDS"] + 1):
                sel = fold_ids == f
                mrows.append(
                    dict(model="ENSEMBLE", fold=f, **evaluate(y[sel], ens_oof[sel]))
                )
            pooled_ens = evaluate(y, ens_oof)
            mrows.append(dict(model="ENSEMBLE", fold="pooled", **pooled_ens))
            fm = pd.DataFrame(mrows)
            threshold = float(pooled_ens["threshold_max_tss"])

            held_final = held.copy()
            for m in ok_models + ["ENSEMBLE"]:
                held_final[m] = oof[m]
            save_csv(
                held_final,
                csv_dir / "06_heldout_predictions.csv",
                "held-out predictions (final)",
            )
            save_csv(fm, csv_dir / "07_cv_metrics_per_fold.csv", "CV metrics per fold")
            perf = (
                fm[fm.fold != "pooled"]
                .groupby("model")[
                    ["auc", "tss", "sensitivity", "specificity", "brier", "boyce"]
                ]
                .agg(["mean", "std"])
            )
            perf.columns = ["_".join(c) for c in perf.columns]
            perf = perf.reset_index().merge(
                fm[fm.fold == "pooled"][
                    ["model", "auc", "tss", "threshold_max_tss"]
                ].rename(columns={"auc": "pooled_auc", "tss": "pooled_tss"}),
                on="model",
            )
            save_csv(perf, csv_dir / "07a_cv_metrics_summary.csv", "CV metrics summary")
            save_csv(
                pd.DataFrame(
                    dict(
                        threshold_type=["max_TSS_pooled_heldout_ensemble"],
                        value=[threshold],
                    )
                ),
                csv_dir / "08a_ensemble_threshold.csv",
                "ensemble threshold",
            )
            log(
                "\n"
                + perf[
                    [
                        "model",
                        "auc_mean",
                        "auc_std",
                        "tss_mean",
                        "boyce_mean",
                        "pooled_auc",
                    ]
                ]
                .round(3)
                .to_string(index=False)
            )
            log(
                "\n"
                + weights[["model", "mean_fold_auc", "weight"]]
                .round(4)
                .to_string(index=False)
            )
            log(
                "NOTE: ensemble weights are derived from the same held-out predictions used to "
                "score the ensemble, so ensemble CV metrics are slightly optimistic."
            )

        # -------------------------------------------------------------- final models
        with stage("final model fitting (all data)", out):
            fitted, final_log = {}, []
            res = joblib.Parallel(n_jobs=min(n_jobs, len(ok_models)))(
                joblib.delayed(_final_task)(
                    m, X, y, hp_of(m), est_over.get(m, {}), cfg["SEED"] + 99999
                )
                for m in ok_models
            )
            for m, model, err, secs in res:
                final_log.append(dict(model=m, seconds=round(secs, 2), error=err))
                if err:
                    log(f"final {m} FAILED: {err}")
                else:
                    fitted[m] = model
                    joblib.dump(
                        model.export(feats),
                        mdl_dir / f"model_{m}_{cfg['RUN_LABEL']}.joblib",
                        compress=3,
                    )
                    log(f"saved final model {m} ({secs:,.1f}s)")
            save_csv(
                pd.DataFrame(final_log),
                csv_dir / "08b_final_fit_log.csv",
                "final fit log",
            )
            if not fitted:
                raise RuntimeError("No final model could be fitted.")
            # re-normalise weights over the models that actually fitted
            weights = weights[weights.model.isin(fitted)].copy()
            weights["weight"] = weights.raw_weight / weights.raw_weight.sum()
            save_csv(
                weights, csv_dir / "08_ensemble_weights.csv", "ensemble weights (final)"
            )

        # -------------------------------------------------------------- rasters
        with stage("raster prediction (ensemble + per-model)", out):
            paths, rsum = predict_rasters(
                stack, aoi_geom, fitted, weights, threshold, feats, cfg, rst_dir
            )
            log(
                "\n"
                + rsum[["raster", "valid_cells", "min", "mean", "max"]]
                .round(4)
                .to_string(index=False)
            )

        # -------------------------------------------------------------- report
        with stage("reproducibility report", out):
            ctx = dict(
                t0=t_run,
                started_utc=started,
                raster_files=files,
                features=feats,
                dropped_const=dropped_const,
                dropped_cor=dropped_cor,
                n_pres=n_pres,
                n_bg=n_bg,
                weights=weights.to_dict("records"),
                threshold=threshold,
                effective_hp=effective_hp,
            )
            report = write_reproducibility_report(cfg, ctx, rep_dir)
    finally:
        stack.close()

    log(f"ALL DONE in {(time.time() - t_run) / 60:,.1f} min. Outputs: {out}")
    return dict(
        output_dir=str(out),
        rasters={k: str(v) for k, v in paths.items()},
        cv_summary=perf,
        weights=weights,
        report=report,
    )


def _joblib_has_generator():
    try:
        return tuple(int(x) for x in joblib.__version__.split(".")[:2]) >= (1, 3)
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":  # also True when run inside a Jupyter cell
    RESULTS = main()
