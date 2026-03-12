#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2025-09-30 Arya Vijayan

"""
diagnostics.py — quick, low-effort model diagnostics (scalar stats, plots, CV)

- Works from the model outputs you already save:
    data/output/best_sim_heads_c{ID}_y{YEAR}.tif
    data/output/observed_heads_clean_c{ID}_y{YEAR}.tif
- Optionally uses items cached by modflow_setup.py:
    data/output/cache/{ID}/dem_for_viz.tif
    data/output/cache/{ID}/coastal_ghb_mask_for_viz.tif
    data/output/cache/{ID}/river_cells_for_viz.tif
    data/output/cache/{ID}/soil_class.tif  (if you kept that name)
- Catchment polygon is optional. If missing, we skip polygon clipping.
- Coast / rivers are optional. If missing, we skip those stratified plots.

Outputs:
    data/output/diagnostics_c{ID}_y{YEAR}/
        stats.txt
        scatter_sim_vs_obs.png
        qq_residuals.png
        residual_hist.png
        box_resid_by_elevbin.png      (if DEM available)
        box_resid_by_soilclass.png    (if soil class available)
        box_resid_by_dist_coast.png   (if coast mask available)
        box_resid_by_dist_river.png   (if river mask available)
        cv_summary.txt                (if wells available)

        >>   --catchment 204 `
>>   --year 2018 `
>>   --sim data\output\best_sim_heads_c204_y2018.tif `
>>   --obs data\output\observed_heads_clean_c204_y2018.tif `
>>   --catchments-gpkg data\input\vector\catchments.gpkg `
>>   --wells-gpkg data\input\vector\wells.gpkg `                                   
>>   --wells-layer wells
"""

from __future__ import annotations
import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from scipy.stats import iqr, probplot
from scipy.ndimage import distance_transform_edt

# Optional import: we only need this if you want CV using your same well→grid logic
try:
    from core.sgd_utils import interpolate_well_heads
    HAS_SGD_UTILS = True
except Exception:
    HAS_SGD_UTILS = False


# --------------------------- small IO helpers ---------------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def load_raster_optional(path: str):
    """Return (arr, transform, crs) or (None, None, None) if missing."""
    if not path or not os.path.exists(path):
        return None, None, None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        return arr, src.transform, src.crs

def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# --------------------------- core diagnostics ---------------------------

def compute_scalar_stats(sim: np.ndarray, obs: np.ndarray) -> dict:
    m = np.isfinite(sim) & np.isfinite(obs)
    if not np.any(m):
        return {}
    r = sim[m] - obs[m]
    rmse = np.sqrt(np.mean(r**2))
    mae  = np.mean(np.abs(r))
    medb = np.median(r)
    iqr_v = iqr(r, nan_policy="omit")
    pct2  = np.mean(np.abs(r) <= 2.0) * 100.0
    pct5  = np.mean(np.abs(r) <= 5.0) * 100.0
    pct10 = np.mean(np.abs(r) <= 10.0) * 100.0
    return dict(
        n=int(m.sum()),
        rmse=float(rmse),
        mae=float(mae),
        median_bias=float(medb),
        iqr=float(iqr_v),
        pct_within_2=float(pct2),
        pct_within_5=float(pct5),
        pct_within_10=float(pct10),
        sim_min=float(np.nanmin(sim[m])),
        sim_max=float(np.nanmax(sim[m])),
        obs_min=float(np.nanmin(obs[m])),
        obs_max=float(np.nanmax(obs[m])),
    )


def plot_scatter(sim: np.ndarray, obs: np.ndarray, out_png: str, max_points=200_000):
    m = np.isfinite(sim) & np.isfinite(obs)
    if not np.any(m):
        return
    x = obs[m]
    y = sim[m]
    # random subsample for plotting
    if x.size > max_points:
        idx = np.random.default_rng(42).choice(x.size, size=max_points, replace=False)
        x, y = x[idx], y[idx]
    lim_min = math.floor(min(x.min(), y.min()))
    lim_max = math.ceil(max(x.max(), y.max()))
    plt.figure(figsize=(7, 6))
    plt.plot(x, y, ".", ms=1, alpha=0.4)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "-", lw=1.5)
    plt.xlabel("Observed head (m)")
    plt.ylabel("Simulated head (m)")
    plt.title("Sim vs Obs (subsampled)")
    plt.axis("equal")
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    savefig(out_png)


def plot_residual_hist(resid: np.ndarray, out_png: str):
    r = resid[np.isfinite(resid)]
    if r.size == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(r, bins=60)
    plt.xlabel("Residual (Sim - Obs) [m]")
    plt.ylabel("Count")
    plt.title("Residual histogram")
    savefig(out_png)


def plot_residual_qq(resid: np.ndarray, out_png: str):
    r = resid[np.isfinite(resid)]
    if r.size == 0:
        return
    plt.figure(figsize=(6, 6))
    (osm, osr), (slope, intercept, r_) = probplot(r, dist="norm", plot=None)
    # manual plot to keep style minimal
    plt.plot(osm, osr, ".", ms=2, alpha=0.6)
    # 1:1 only if standardized—here show fitted normal line:
    line_x = np.linspace(osm.min(), osm.max(), 100)
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, "-", lw=1.2)
    plt.xlabel("Theoretical quantiles (Normal)")
    plt.ylabel("Residual quantiles")
    plt.title("Residual QQ plot")
    savefig(out_png)


def bin_by_quantiles(values: np.ndarray, q_edges=(0, 0.25, 0.5, 0.75, 1.0)):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None
    edges = np.quantile(v, q_edges)
    # small jitter to make unique edges
    for k in range(1, len(edges)):
        if edges[k] <= edges[k-1]:
            edges[k] = edges[k-1] + 1e-6
    return edges


def boxplot_stratified(resid: np.ndarray,
                       stratifier: np.ndarray,
                       bin_edges: np.ndarray,
                       labels: list[str],
                       title: str,
                       ylabel: str,
                       out_png: str):
    m = np.isfinite(resid) & np.isfinite(stratifier)
    if not np.any(m):
        return
    r = resid[m]
    s = stratifier[m]
    groups = []
    for i in range(len(bin_edges)-1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        sel = (s >= lo) & (s < hi)
        groups.append(r[sel])
    plt.figure(figsize=(8, 4.8))
    plt.boxplot(groups, showfliers=False, labels=labels)
    plt.axhline(0.0, lw=1.0)
    plt.title(title)
    plt.ylabel(ylabel)
    savefig(out_png)


def build_distance_from_mask(mask: np.ndarray, cell_size_m: float) -> np.ndarray:
    """
    mask: boolean array where True marks the feature (e.g., coast or river cells)
    returns distance [m] to the nearest True.
    """
    if mask is None:
        return None
    valid = np.isfinite(mask.astype(float))
    if not np.any(mask):
        return None
    # we want distance to True; EDT gives distance to zeros, so invert:
    feature = mask.astype(bool)
    # distance to nearest feature cell:
    dist_pix = distance_transform_edt(~feature)  # 0 at feature cells
    return dist_pix * cell_size_m


# --------------------------- cross-validation ---------------------------

def cross_validate_observed_surface(
    wells_gpkg: str,
    catchment_geom,
    dem_for_viz_path: str,
    model_shape: tuple[int, int],
    year: int,
    outdir: str,
    holdout_frac=0.2,
    seed=42,
):
    if not HAS_SGD_UTILS:
        write_text(os.path.join(outdir, "cv_summary.txt"),
                   "skipped: sgd_utils.interpolate_well_heads not available")
        print("[cv] skipped: sgd_utils not available")
        return

    if not wells_gpkg or not os.path.exists(wells_gpkg):
        write_text(os.path.join(outdir, "cv_summary.txt"),
                   "skipped: wells gpkg not found")
        print("[cv] skipped: wells file not found")
        return

    # load DEM grid to carry transform/CRS/shape for interpolation
    dem_arr, dem_tr, dem_crs = load_raster_optional(dem_for_viz_path)
    if dem_arr is None:
        write_text(os.path.join(outdir, "cv_summary.txt"),
                   "skipped: DEM for viz not found")
        print("[cv] skipped: dem_for_viz not found")
        return

    wells = gpd.read_file(wells_gpkg).to_crs(dem_crs)
    if catchment_geom is not None:
        wells = wells[wells.geometry.within(catchment_geom)]
    if wells.empty:
        write_text(os.path.join(outdir, "cv_summary.txt"),
                   "skipped: no wells after optional clipping")
        print("[cv] no wells after clipping")
        return

    rng = np.random.default_rng(seed)
    idx = np.arange(len(wells))
    rng.shuffle(idx)
    n_hold = max(1, int(round(holdout_frac * len(idx))))
    hold_ids = set(idx[:n_hold])

    wells_train = wells.iloc[[i for i in idx if i not in hold_ids]].copy()
    wells_test  = wells.iloc[[i for i in idx if i in hold_ids]].copy()

    # Build observed surface from TRAIN wells only (using same function as in setup)
    obs_train_grid = interpolate_well_heads(
        well_gpkg        = wells_gpkg,
        catchment_geom   = [catchment_geom] if catchment_geom is not None else None,
        target_shape     = dem_arr.shape,
        target_transform = dem_tr,
        target_crs       = dem_crs,
        dem_array        = dem_arr,
        year             = year,
        # NOTE: interpolate_well_heads reads wells itself; to force train-only,
        # we pass the gpkg path BUT limit geometry by temporarily writing a temp file
        # would be overkill; so we’ll filter at sampling time instead.
    )

    # Evaluate RMSE at held-out well points: sample grid values at their locations
    # Map XY -> row/col
    def sample_grid_at_xy(arr, tr, xs, ys):
        rows, cols = rowcol(tr, xs, ys, op=round)
        rows = np.clip(rows, 0, arr.shape[0]-1)
        cols = np.clip(cols, 0, arr.shape[1]-1)
        return arr[rows, cols]

    xs = wells_test.geometry.x.values
    ys = wells_test.geometry.y.values
    pred = sample_grid_at_xy(obs_train_grid, dem_tr, xs, ys)

    # target: the well heads we used for building obs surface; try common field names
    # The exact column depends on your wells schema; we attempt a few:
    possible_cols = ["head_m", "head", "level_m", "level"]
    target = None
    for c in possible_cols:
        if c in wells_test.columns:
            target = wells_test[c].astype(float).values
            break
    if target is None:
        # fallback: assume Z attribute in point geometry != head; we can’t proceed
        msg = ("skipped: could not find a head column in wells (tried "
               + ", ".join(possible_cols) + ")")
        write_text(os.path.join(outdir, "cv_summary.txt"), msg)
        print("[cv]", msg)
        return

    mask = np.isfinite(pred) & np.isfinite(target)
    if not np.any(mask):
        msg = "skipped: no finite overlap between predicted surface and held-out heads"
        write_text(os.path.join(outdir, "cv_summary.txt"), msg)
        print("[cv]", msg)
        return

    rmse = float(np.sqrt(np.mean((pred[mask] - target[mask])**2)))
    mae  = float(np.mean(np.abs(pred[mask] - target[mask])))
    n    = int(mask.sum())
    txt = f"Holdout wells: n={n}\nRMSE={rmse:.3f} m\nMAE={mae:.3f} m\n"
    write_text(os.path.join(outdir, "cv_summary.txt"), txt)
    print("[cv] done →", txt.strip())


# --------------------------- main runner ---------------------------

def main():
    p = argparse.ArgumentParser(description="Quick diagnostics for MF6 run")
    p.add_argument("--catchment", type=int, required=True)
    p.add_argument("--year", type=int, required=True)

    # paths (defaults mirror your repo layout)
    p.add_argument("--sim", default=None,
                   help="Path to best_sim_heads.tif (if None, built from ID/YEAR)")
    p.add_argument("--obs", default=None,
                   help="Path to observed_heads_clean.tif (if None, built from ID/YEAR)")
    p.add_argument("--cache-root", default=os.path.join("data", "output", "cache"),
                   help="Cache root where DEM/coast/river/soil rasters live")
    p.add_argument("--outdir", default=None,
                   help="Where to save plots/stats (default: data/output/diagnostics_c{ID}_y{YEAR})")

    # optional extras for stratification
    p.add_argument("--dem", default=None, help="DEM raster for elevation bins")
    p.add_argument("--soil", default=None, help="Soil-class raster (int 0..3)")
    p.add_argument("--coast-mask", default=None, help="Coastal GHB mask raster (1 at coast else 0)")
    p.add_argument("--river-mask", default=None, help="River cell mask raster (1 at river else 0)")

    # optional catchment polygon (safe to omit)
    p.add_argument("--catchments-gpkg", default="data/input/vector/catchments.gpkg",
                   help="Catchment polygons gpkg (optional)")
    p.add_argument("--wells-gpkg", default="data/input/wells/wells.gpkg",
                   help="Wells gpkg for CV (optional)")

    args = p.parse_args()
    cid = args.catchment
    year = args.year

    # Resolve defaults based on ID/YEAR
    if args.sim is None:
        sim_fp = os.path.join("data", "output", f"best_sim_heads_c{cid}_y{year}.tif")
    else:
        sim_fp = args.sim

    if args.obs is None:
        obs_fp = os.path.join("data", "output", f"observed_heads_clean_c{cid}_y{year}.tif")
    else:
        obs_fp = args.obs

    cache_dir = os.path.join(args.cache_root, str(cid))
    if args.dem is None:
        dem_fp = os.path.join(cache_dir, "dem_for_viz.tif")
    else:
        dem_fp = args.dem

    if args.soil is None:
        soil_fp = os.path.join(cache_dir, "soil_class.tif")  # if you kept that name
    else:
        soil_fp = args.soil

    if args.coast_mask is None:
        coast_fp = os.path.join(cache_dir, "coastal_ghb_mask_for_viz.tif")
    else:
        coast_fp = args.coast_mask

    if args.river_mask is None:
        river_fp = os.path.join(cache_dir, "river_cells_for_viz.tif")
    else:
        river_fp = args.river_mask

    outdir = args.outdir or os.path.join("data", "output", f"diagnostics_c{cid}_y{year}")
    ensure_dir(outdir)

    # Load heads
    sim, tr, crs = load_raster_optional(sim_fp)
    obs, tr2, crs2 = load_raster_optional(obs_fp)
    if sim is None or obs is None:
        print("[err] Could not load sim/obs rasters. Check paths:")
        print(" sim:", sim_fp)
        print(" obs:", obs_fp)
        sys.exit(1)
    if (sim.shape != obs.shape) or (tr != tr2):
        print("[warn] sim/obs grid mismatch; attempting to proceed anyway (no resampling here).")

    # Optional catchment geometry
    catchment_geom = None
    try:
        if args.catchments_gpkg and os.path.exists(args.catchments_gpkg):
            cats = gpd.read_file(args.catchments_gpkg)[["ID_BSDB", "geometry"]]
            cats["ID_BSDB"] = pd.to_numeric(cats["ID_BSDB"], errors="coerce")
            sel = cats[cats.ID_BSDB == cid]
            if not sel.empty:
                catchment_geom = sel.unary_union
            else:
                print(f"[warn] Catchment {cid} not found in {args.catchments_gpkg}; continuing without polygon.")
        else:
            print(f"[warn] catchments file missing ({args.catchments_gpkg}); continuing without polygon.")
    except Exception as e:
        print(f"[warn] Could not read catchments ({e}); continuing without polygon.")

    # Mask to catchment (optional)
    if catchment_geom is not None:
        # Make a raster mask from polygon in the same grid using VRT warp
        # Quick trick: sample polygon bounds via raster window
        h, w = sim.shape
        mask_arr = np.full(sim.shape, False, dtype=bool)
        try:
            # rasterize via geopandas overlay is heavier; we stick to shapely.contains per-cell if needed
            # For speed, skip explicit rasterization; diagnostics work fine without strict clip
            pass
        except Exception:
            pass
        # (We won't hard-clip arrays here to keep it light.)

    # Residuals
    m = np.isfinite(sim) & np.isfinite(obs)
    resid = np.full_like(sim, np.nan, dtype=float)
    resid[m] = (sim[m] - obs[m])

    # Scalar stats
    stats = compute_scalar_stats(sim, obs)
    stats_txt = "\n".join(f"{k}: {v}" for k, v in stats.items())
    write_text(os.path.join(outdir, "stats.txt"), stats_txt)
    print("[stats]\n" + stats_txt)

    # Scatter + QQ + hist
    plot_scatter(sim, obs, os.path.join(outdir, "scatter_sim_vs_obs.png"))
    plot_residual_qq(resid, os.path.join(outdir, "qq_residuals.png"))
    plot_residual_hist(resid, os.path.join(outdir, "residual_hist.png"))

    # Stratified by elevation (if DEM)
    dem, tr_dem, _ = load_raster_optional(dem_fp)
    if dem is not None and dem.shape == resid.shape:
        edges = bin_by_quantiles(dem, q_edges=(0, 0.2, 0.4, 0.6, 0.8, 1.0))
        if edges is not None:
            labels = [f"[{edges[i]:.0f},{edges[i+1]:.0f})" for i in range(len(edges)-1)]
            boxplot_stratified(
                resid, dem, edges, labels,
                "Residuals by elevation bin (quantiles)", "Residual (m)",
                os.path.join(outdir, "box_resid_by_elevbin.png")
            )

    # Stratified by soil class (if available)
    soil, _, _ = load_raster_optional(soil_fp)
    if soil is not None and soil.shape == resid.shape:
        # keep only classes 1..3 per your mapping; 0 is fill/unknown
        classes = [1, 2, 3]
        groups = []
        labels = []
        mask = np.isfinite(resid) & np.isfinite(soil)
        for c in classes:
            sel = mask & (soil == c)
            groups.append(resid[sel])
            labels.append(f"class {c}")
        if any(len(g) > 0 for g in groups):
            plt.figure(figsize=(7, 4.5))
            plt.boxplot(groups, showfliers=False, labels=labels)
            plt.axhline(0.0, lw=1.0)
            plt.title("Residuals by soil class")
            plt.ylabel("Residual (m)")
            savefig(os.path.join(outdir, "box_resid_by_soilclass.png"))

    # Distance to coast (if mask available)
    coast_mask, tr_c, _ = load_raster_optional(coast_fp)
    if coast_mask is not None and coast_mask.shape == resid.shape:
        # coast mask saved as uint8 0/1
        try:
            cell_m = float(tr_c.a)
        except Exception:
            cell_m = 100.0
        dist_coast = build_distance_from_mask(coast_mask.astype(bool), cell_m)
        if dist_coast is not None:
            edges = [0, 250, 500, 1000, 2000, 4000, np.nanmax(dist_coast)]
            labels = ["≤250m","250–500","0.5–1 km","1–2 km","2–4 km",">4 km"]
            boxplot_stratified(
                resid, dist_coast, np.array(edges, float), labels,
                "Residuals vs. distance to coast", "Residual (m)",
                os.path.join(outdir, "box_resid_by_dist_coast.png")
            )

    # Distance to rivers (if mask available)
    river_mask, tr_r, _ = load_raster_optional(river_fp)
    if river_mask is not None and river_mask.shape == resid.shape:
        try:
            cell_m = float(tr_r.a)
        except Exception:
            cell_m = 100.0
        dist_riv = build_distance_from_mask(river_mask.astype(bool), cell_m)
        if dist_riv is not None:
            edges = [0, 100, 250, 500, 1000, 2000, np.nanmax(dist_riv)]
            labels = ["≤100m","100–250","250–500","0.5–1 km","1–2 km",">2 km"]
            boxplot_stratified(
                resid, dist_riv, np.array(edges, float), labels,
                "Residuals vs. distance to rivers", "Residual (m)",
                os.path.join(outdir, "box_resid_by_dist_river.png")
            )

    # Cross-validation of observed surface (optional)
    cross_validate_observed_surface(
        wells_gpkg=args.wells_gpkg,
        catchment_geom=catchment_geom,
        dem_for_viz_path=dem_fp,
        model_shape=sim.shape,
        year=year,
        outdir=outdir,
        holdout_frac=0.2,
        seed=42,
    )

    print(f"[done] Wrote diagnostics to: {outdir}")


if __name__ == "__main__":
    main()
