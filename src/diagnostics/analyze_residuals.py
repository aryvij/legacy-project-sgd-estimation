# src/analyze_residuals.py
import os
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

def read_tif(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(float)
        tr = ds.transform
        crs = ds.crs
        nodata = ds.nodata
    if np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr, tr, crs

def nse(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    if not np.any(m): return np.nan
    num = np.nansum((sim[m]-obs[m])**2)
    den = np.nansum((obs[m]-np.nanmean(obs[m]))**2)
    return 1.0 - (num/den if den>0 else np.nan)

def global_metrics(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    if not np.any(m):
        return dict(valid=0, rmse=np.nan, mae=np.nan, bias=np.nan, p10=np.nan, p90=np.nan, nse=np.nan)
    res = sim[m] - obs[m]
    return dict(
        valid=int(m.sum()),
        rmse=float(np.sqrt(np.nanmean(res**2))),
        mae=float(np.nanmean(np.abs(res))),
        bias=float(np.nanmedian(res)),
        p10=float(np.nanpercentile(res, 10)),
        p90=float(np.nanpercentile(res, 90)),
        nse=float(nse(sim, obs))
    )

def binned_stats(residual, by, bins, labels):
    out = []
    for (lo, hi), lab in zip(zip(bins[:-1], bins[1:]), labels):
        m = (by >= lo) & (by < hi) & np.isfinite(residual)
        if not np.any(m):
            out.append([lab, 0, np.nan, np.nan, np.nan])
            continue
        r = residual[m]
        out.append([lab, int(m.sum()), float(np.nanmedian(r)), float(np.nanpercentile(r,10)), float(np.nanpercentile(r,90))])
    return pd.DataFrame(out, columns=["bin","n","median_res","p10_res","p90_res"])

def main(catch_id, year, output_dir=None):
    base = output_dir or "data/output"
    cache = os.path.join(base, "cache", str(catch_id))
    # Inputs you already produced
    sim_tif = os.path.join(base, f"best_sim_heads_c{catch_id}_y{year}.tif")
    obs_tif = os.path.join(base, f"observed_heads_clean_c{catch_id}_y{year}.tif")
    dem_tif = os.path.join(cache, "dem_for_viz.tif")
    riv_mask_tif = os.path.join(cache, str(catch_id), "river_cells_for_viz.tif")  # note: your logs show cache\204\river_cells_for_viz.tif
    ghb_mask_tif = os.path.join(cache, str(catch_id), "coastal_ghb_mask_for_viz.tif")

    # In your logs, river/coast masks are in data/output/cache/204/*.tif (no subfolder).
    # So fall back to those if nested paths don't exist:
    if not os.path.exists(riv_mask_tif):
        riv_mask_tif = os.path.join(cache, "river_cells_for_viz.tif")
    if not os.path.exists(ghb_mask_tif):
        ghb_mask_tif = os.path.join(cache, "coastal_ghb_mask_for_viz.tif")

    sim, tr, _ = read_tif(sim_tif)
    obs, _, _ = read_tif(obs_tif)
    dem, _, _ = read_tif(dem_tif)

    # Residual
    res = sim - obs
    valid = np.isfinite(res)
    print(f"[{year}] valid cells: {valid.sum()}  res min/max: {np.nanmin(res):.2f}…{np.nanmax(res):.2f}")

    # Global metrics
    gm = global_metrics(sim, obs)
    gm_df = pd.DataFrame([gm])
    gm_df.to_csv(os.path.join(base, f"metrics_global_c{catch_id}_y{year}.csv"), index=False)
    print(gm_df)

    # Elevation-binned residuals
    elev_bins = np.array([-1e9, 10, 30, 60, 1e9])
    elev_labels = ["<10","10–30","30–60",">60"]
    elev_stats = binned_stats(res, dem, elev_bins, elev_labels)
    elev_stats.to_csv(os.path.join(base, f"metrics_res_vs_elev_c{catch_id}_y{year}.csv"), index=False)

    # Distances to rivers & coast from masks you already saved
    def dist_from_mask(mask_arr):
        # mask_arr==1 on feature pixels; distance 0 there; NaN where invalid
        m = np.isfinite(mask_arr)
        feat = (mask_arr == 1) & m
        if not np.any(feat):
            return np.full(mask_arr.shape, np.nan)
        # distance in pixels; convert to meters using transform.a (pixel size)
        pixdist = distance_transform_edt(~feat & m)
        # pixel size (assume square)
        px = abs(tr.a)
        return pixdist * px

    # Load masks; if missing, skip those diagnostics
    try:
        riv_mask, _, _ = read_tif(riv_mask_tif)
        dist_riv = dist_from_mask((riv_mask>0).astype(float))
    except Exception:
        dist_riv = np.full(sim.shape, np.nan)

    try:
        ghb_mask, _, _ = read_tif(ghb_mask_tif)
        dist_coast = dist_from_mask((ghb_mask>0).astype(float))
    except Exception:
        dist_coast = np.full(sim.shape, np.nan)

    # Bin by distances
    riv_bins = np.array([0, 250, 500, 2000, 1e9])   # meters
    riv_labels = ["0–0.25 km","0.25–0.5 km","0.5–2 km",">2 km"]
    coast_bins = np.array([0, 500, 2000, 1e9])
    coast_labels = ["0–0.5 km","0.5–2 km",">2 km"]

    # Make sure residuals are valid where distances are valid
    res_riv = np.where(np.isfinite(dist_riv), res, np.nan)
    res_coast = np.where(np.isfinite(dist_coast), res, np.nan)

    riv_stats = binned_stats(res_riv, dist_riv, riv_bins, riv_labels)
    coast_stats = binned_stats(res_coast, dist_coast, coast_bins, coast_labels)

    riv_stats.to_csv(os.path.join(base, f"metrics_res_vs_river_c{catch_id}_y{year}.csv"), index=False)
    coast_stats.to_csv(os.path.join(base, f"metrics_res_vs_coast_c{catch_id}_y{year}.csv"), index=False)

    # Quick residual histogram
    plt.figure(figsize=(6,4))
    plt.hist(res[valid], bins=60)
    plt.title(f"Residual histogram (Sim-Obs) — c{catch_id} y{year}")
    plt.xlabel("Residual (m)"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(base, f"residual_hist_c{catch_id}_y{year}.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--catchment", type=int, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Path to output folder (default: data/output)")
    args = ap.parse_args()
    main(args.catchment, args.year, output_dir=args.output_dir)
