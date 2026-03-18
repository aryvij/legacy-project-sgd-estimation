#!/usr/bin/env python3
# sensitivity_oat.py
# One-at-a-time sensitivity around a calibrated parameter set
# Adds: SGD from CBC, progress counter, ETA, per-run CSV append, retry logic, ASCII-only prints,
#       and cleaning of scratch folders to reduce disk/RAM pressure.

# PowerShell UTF-8 setup (optional, avoids UnicodeEncodeError in the console):
# chcp 65001 > $null
# $env:PYTHONUTF8 = "1"
# $env:PYTHONIOENCODING = "utf-8"
# [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
#
# Example run:
# $log = "data\output\logs\sens_oat_204_2019.log"
# python -u src\sensitivity_oat.py `
#   --catchment 204 `
#   --year 2019 `
#   --mf6 "C:\Users\aryapv\AppData\Local\Programs\mf6.6.1_win64\bin\mf6.exe" `
#   --params data/output/calib_final_params_c204_y2010.json `
#   --outdir data/output | Tee-Object -FilePath $log

import os
# Conservative caps to keep memory usage in check
os.environ.setdefault("GDAL_CACHEMAX", "256")           # MB
os.environ.setdefault("RASTERIO_MAXIMUM_RAM", "512MB")  # e.g., "512MB"

import sys, json, argparse, csv, pathlib, time, shutil
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

from core.modflow_setup import setup_and_run_modflow, load_or_interpolate_obs_heads

_DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_INPUT  = _DEFAULT_ROOT / "data" / "input"
_DEFAULT_OUTPUT = _DEFAULT_ROOT / "data" / "output"

def cache_dir(filepaths, catchment_id: int, year: int) -> str:
    """Per-run cache folder (safe to delete)."""
    return os.path.join(filepaths["output"], "cache", str(catchment_id), str(year))

def safe_rmtree(path: str):
    """Delete a directory tree if it exists (ignore errors)."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        # Best-effort cleanup; do not fail if deletion has locking issues
        pass

def polygon_mask(poly, shape, transform):
    return rasterize(
        [(poly, 1)],
        out_shape=shape, transform=transform,
        fill=0, all_touched=False, dtype="uint8"
    ).astype(bool)

def rmse_vs_obs(sim_heads, obs_heads, mask):
    s = sim_heads.astype(float).copy()
    o = obs_heads.astype(float).copy()
    s[~np.isfinite(s)] = np.nan
    o[~np.isfinite(o)] = np.nan
    m = mask & np.isfinite(s) & np.isfinite(o)
    if m.sum() == 0:
        return np.inf
    return float(np.sqrt(np.nanmean((s[m] - o[m])**2)))

def run_workspace(filepaths, catchment_id: int) -> str:
    # Must match base_ws in modflow_setup.py
    return os.path.join(filepaths["output"], "model_runs", f"mf6_{catchment_id}")

def read_sgd_m3d_from_cbc(run_ws: str, catchment_id: int) -> float:
    """
    Read last-time-step SGD (m3/d) from GHB flows in the CBC.
    MF6 budget convention: positive = INTO the model.
    SGD is water leaving model to sea via GHB, so sum negatives and flip sign.
    """
    import flopy
    cbcfile = os.path.join(run_ws, f"gwf_{catchment_id}.cbc")
    if not os.path.exists(cbcfile):
        return float("nan")
    try:
        cbc = flopy.utils.CellBudgetFile(cbcfile, precision="double")
        for label in ("GHB", b"GHB"):
            recs = cbc.get_data(text=label)
            if recs:
                arr = recs[-1]
                if hasattr(arr, "dtype") and (arr.dtype.names and ("q" in arr.dtype.names)):
                    q = np.asarray(arr["q"], dtype=float)
                else:
                    q = np.asarray(arr, dtype=float)
                sgd = float(-np.nansum(q[q < 0.0]))
                return sgd
    except Exception:
        return float("nan")
    return float("nan")

def run_with_retry(catchment, filepaths, mf6, x, year, coastal_buffer=200.0, max_retries=2):
    """
    Try to run MF6 with a couple of small relaxations if it fails to converge.
    Returns (heads, dem_tr, dem_crs, catch_poly) or raises after retries.
    """
    attempts = []
    # baseline
    attempts.append((x[0], x[1], x[2], x[3], x[4], "baseline"))
    # soften multipliers a bit if needed
    attempts.append((x[0], x[1], x[2]*0.90, x[3], x[4], "riv*0.90"))
    attempts.append((x[0], x[1], x[2], x[3]*0.90, x[4], "ghb*0.90"))

    tried = 0
    for soilK, rockK, riv, ghb, rch, tag in attempts:
        tried += 1
        print(f"[retry {tried}/{min(len(attempts), 1+max_retries)}] trying riv={riv:.3f}, ghb={ghb:.3f}, rch={rch:.3f}")
        try:
            return setup_and_run_modflow(
                catchment_id=catchment, filepaths=filepaths, coastal_buffer=coastal_buffer, mf6_exe=mf6,
                soilK_multiplier=soilK, rockK_multiplier=rockK,
                riv_cond_multiplier=riv, ghb_cond_multiplier=ghb,
                recharge_year=year, rch_multiplier=rch
            )
        except RuntimeError:
            if tried >= 1 + max_retries:
                raise
            # otherwise loop and try next variant
    # If we got here, raise
    raise RuntimeError("MODFLOW run failed after retries")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catchment", "-c", type=int, required=True)
    ap.add_argument("--year", "-y", type=int, required=True, help="Validation year to evaluate sensitivities")
    ap.add_argument("--mf6", required=True, help="Path to mf6 executable")
    ap.add_argument("--params", required=True, help="Path to calibrated params JSON")
    ap.add_argument("--deltas", default="-0.2,-0.1,0.1,0.2",
                    help="Comma-separated fractional changes, e.g. -0.2,-0.1,0,0.1,0.2")
    ap.add_argument("--outdir", default=None, help="Output folder (default: <output-dir>)")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Path to input data folder (default: <project>/data/input)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Path to output folder (default: <project>/data/output)")
    args = ap.parse_args()

    INPUT  = pathlib.Path(args.data_root)  if args.data_root  else _DEFAULT_INPUT
    OUTPUT = pathlib.Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT
    DATA   = OUTPUT.parent
    if args.outdir is None:
        args.outdir = str(OUTPUT)

    with open(args.params, "r") as f:
        pcal = json.load(f)

    # Parameter order matches setup_and_run_modflow signature multipliers
    x0 = np.array([
        pcal["soilK_multiplier"],
        pcal["rockK_multiplier"],
        pcal["riv_cond_multiplier"],
        pcal["ghb_cond_multiplier"],
        pcal["rch_multiplier"],
    ], dtype=float)

    # Filepaths (use yearly recharge tif)
    filepaths = {
        "dem"         : INPUT / "dem" / "elevation_sweden.tif",
        "catchment"   : INPUT / "shapefiles" / "catchment" / "bsdbs.shp",
        "recharge"    : OUTPUT / "recharge_yearly" / f"recharge_egdi_gldas_{args.year}.tif",
        "soil_perm"   : INPUT / "aquifer_data" / "genomslapplighet" / "genomslapplighet.gpkg",
        "soil_depth"  : INPUT / "aquifer_data" / "jorddjupsmodell" / "jorddjupsmodell_10x10m.tif",
        "conductivity": INPUT / "other_rasters" / "hydraulic_conductivity.tif",
        "sea_level"   : INPUT / "sea_level" / "yearly_average_sea_level.csv",
        "coast"       : INPUT / "shapefiles" / "coast_line" / "coastline.shp",
        "wells"       : INPUT / "well_data" / "brunnar.gpkg",
        "rivers"      : INPUT / "shapefiles" / "surface_water" / "Surface_water" / "hl_riks.shp",
        "lakes"       : INPUT / "shapefiles" / "surface_water" / "scandinavian_waters_polygons" / "scandinavian_waters_polygons.shp",
        "output"      : OUTPUT,
    }
    # Ensure plain strings (FloPy sometimes prefers str over Path)
    filepaths = {k: str(v) for k, v in filepaths.items()}

    # Clean scratch before baseline (prevents stale/corrupt temp files)
    safe_rmtree(run_workspace(filepaths, args.catchment))
    safe_rmtree(cache_dir(filepaths, args.catchment, args.year))

    # Baseline run (with retry)
    heads0, dem_tr, dem_crs, catch_poly = run_with_retry(
        catchment=args.catchment, filepaths=filepaths, mf6=args.mf6, x=x0, year=args.year, max_retries=2
    )

    # Observed heads on same grid
    obs = load_or_interpolate_obs_heads(
        well_path   = filepaths["wells"],
        dem_path    = filepaths["dem"],
        catch_poly  = catch_poly,
        year        = args.year,
        dem_tr      = dem_tr,
        dem_crs     = dem_crs,
        model_shape = heads0.shape,
        cache_dir   = os.path.join(filepaths["output"], "cache", str(args.catchment), str(args.year)),
    )

    # Comparison mask: catchment polygon only (simple & consistent)
    mask = polygon_mask(catch_poly, heads0.shape, dem_tr)
    base_rmse = rmse_vs_obs(heads0, obs, mask)

    # Baseline SGD from CBC
    base_ws  = run_workspace(filepaths, args.catchment)
    base_sgd = read_sgd_m3d_from_cbc(base_ws, args.catchment)
    print(f"[baseline] RMSE={base_rmse:.3f} | SGD={base_sgd:.1f} m3/d")

    # Deltas and loop bookkeeping
    deltas = [float(s) for s in args.deltas.split(",") if s.strip() != ""]
    names  = ["soilK_mult","rockK_mult","riv_mult","ghb_mult","rch_mult"]
    total = len(names) * len(deltas)
    done = 0
    start_time = time.time()

    # CSV path
    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, f"sens_oat_c{args.catchment}_y{args.year}.csv")

    # Sweep parameters
    for i, name in enumerate(names):
        for d in deltas:
            x = x0.copy()
            x[i] = x0[i] * (1.0 + d)

            try:
                # Clean scratch before each sensitivity run (keeps RAM/disk usage low)
                safe_rmtree(run_workspace(filepaths, args.catchment))
                # If you need maximum safety (slower), also clear the year cache:
                # safe_rmtree(cache_dir(filepaths, args.catchment, args.year))

                heads, *_ = run_with_retry(
                    catchment=args.catchment, filepaths=filepaths, mf6=args.mf6, x=x, year=args.year, max_retries=2
                )
                score = rmse_vs_obs(heads, obs, mask)
                ws = run_workspace(filepaths, args.catchment)
                sgd_m3d = read_sgd_m3d_from_cbc(ws, args.catchment)
            except Exception as e:
                score = float("nan")
                sgd_m3d = float("nan")
                print(f"[OAT] run failed for {name}, d={d:+.2f}: {e}")

            row = {
                "param": name, "delta_frac": d, "value": x[i],
                "RMSE": score,
                "dRMSE": (score - base_rmse) if np.isfinite(score) and np.isfinite(base_rmse) else np.nan,
                "RMSE_rel": (score / base_rmse) if (np.isfinite(score) and np.isfinite(base_rmse) and base_rmse>0) else np.nan,
                "SGD_m3d": sgd_m3d,
                "dSGD_m3d": (sgd_m3d - base_sgd) if (np.isfinite(sgd_m3d) and np.isfinite(base_sgd)) else np.nan,
                "SGD_rel":  (sgd_m3d / base_sgd)  if (np.isfinite(sgd_m3d) and np.isfinite(base_sgd) and base_sgd>0) else np.nan,
            }

            # progress + ETA
            done += 1
            elapsed = time.time() - start_time
            eta = (elapsed / max(done,1)) * (total - done)
            print(f"[OAT] ({done}/{total}) {name:10s} d={d:+.2f} -> RMSE={row['RMSE']:.3f} (d={row['dRMSE']:+.3f}) | "
                  f"SGD={row['SGD_m3d']:.0f} m3/d (d={row['dSGD_m3d']:+.0f}) | ETA ~ {int(eta//60)}m {int(eta%60)}s")

            # append to CSV right away
            header = list(row.keys())
            write_header = (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0)
            with open(out_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                if write_header:
                    w.writeheader()
                w.writerow(row)

    # Optional summary plots
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(out_csv)
        df20 = df[np.isclose(np.abs(df["delta_frac"]), 0.2, atol=1e-9)]

        # Plot 1: mean |dRMSE| by param
        agg_rmse = df20.groupby("param")["dRMSE"].apply(lambda s: np.nanmean(np.abs(s))).reset_index()
        agg_rmse = agg_rmse.sort_values("dRMSE", ascending=False)
        plt.figure(figsize=(6,4))
        plt.bar(agg_rmse["param"], agg_rmse["dRMSE"])
        plt.ylabel("Mean |dRMSE| (+/-20%)")
        plt.title(f"OAT Sensitivity (Catchment {args.catchment}, {args.year})")
        plt.tight_layout()
        out_png1 = os.path.join(args.outdir, f"sens_oat_dRMSE_c{args.catchment}_y{args.year}.png")
        plt.savefig(out_png1, dpi=200)
        plt.close()
        print(f"Saved OAT dRMSE plot: {out_png1}")

        # Plot 2: mean |dSGD| by param
        if "dSGD_m3d" in df20.columns:
            agg_sgd = df20.groupby("param")["dSGD_m3d"].apply(lambda s: np.nanmean(np.abs(s))).reset_index()
            agg_sgd = agg_sgd.sort_values("dSGD_m3d", ascending=False)
            plt.figure(figsize=(6,4))
            plt.bar(agg_sgd["param"], agg_sgd["dSGD_m3d"])
            plt.ylabel("Mean |dSGD| m3/d (+/-20%)")
            plt.title(f"OAT Sensitivity of SGD (Catchment {args.catchment}, {args.year})")
            plt.tight_layout()
            out_png2 = os.path.join(args.outdir, f"sens_oat_dSGD_c{args.catchment}_y{args.year}.png")
            plt.savefig(out_png2, dpi=200)
            plt.close()
            print(f"Saved OAT dSGD plot: {out_png2}")

    except Exception as e:
        print(f"(summary plots skipped: {e})")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
