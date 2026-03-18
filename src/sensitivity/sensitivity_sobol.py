#!/usr/bin/env python3
# sensitivity_sobol.py
# Global Sobol sensitivity using SALib for user-selected parameters.
# - Choose params with --use-params, e.g. soilK,rockK,rch
# - SGD (optional, --sgd ghb): sum of NEGATIVE GHB flows (last step), sign flipped -> +m3/d (same as OAT)
# - Retry wrapper for MF6, optional workspace cleaning (--no-clean)
# - Writes samples+outputs CSV and Sobol indices JSON
# - Requires: SALib, numpy, rasterio, flopy, your modflow_setup module

import os, sys, json, argparse, pathlib, time, shutil
import numpy as np

# Keep GDAL/RasterIO memory tame
os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("RASTERIO_MAXIMUM_RAM", "512MB")

# --- SALib: prefer sobol sampler, fallback to saltelli if older SALib ---
try:
    from SALib.sample import sobol as sobol_sample_mod  # SALib >= 1.5
    def sobol_sample(problem, N, calc_second_order, seed=None):
        return sobol_sample_mod.sample(problem, N=N, calc_second_order=calc_second_order, seed=seed)
    sobol_sampler_name = "sobol"
except Exception:
    from SALib.sample import saltelli as sobol_sample_mod  # SALib <= 1.4
    def sobol_sample(problem, N, calc_second_order, seed=None):
        # saltelli.sample has no seed kwarg
        return sobol_sample_mod.sample(problem, N, calc_second_order=calc_second_order)
    sobol_sampler_name = "saltelli"

from SALib.analyze import sobol
import rasterio
from rasterio.features import rasterize

from core.modflow_setup import setup_and_run_modflow, load_or_interpolate_obs_heads

_DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_INPUT  = _DEFAULT_ROOT / "data" / "input"
_DEFAULT_OUTPUT = _DEFAULT_ROOT / "data" / "output"

# ------------------- utilities -------------------
def polygon_mask(poly, shape, transform):
    """Rasterize polygon to boolean mask with given shape/transform."""
    return rasterize(
        [(poly, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

def rmse_vs_obs(sim_heads, obs_heads, mask):
    s = sim_heads.astype(float).copy()
    o = obs_heads.astype(float).copy()
    s[~np.isfinite(s)] = np.nan
    o[~np.isfinite(o)] = np.nan
    m = mask & np.isfinite(s) & np.isfinite(o)
    if m.sum() == 0:
        return np.inf
    return float(np.sqrt(np.nanmean((s[m] - o[m]) ** 2)))

def run_workspace(filepaths, catchment_id: int) -> str:
    # must match base_ws in modflow_setup.py
    return os.path.join(filepaths["output"], "model_runs", f"mf6_{catchment_id}")

def safe_rmtree(path: str):
    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def read_sgd_m3d_from_cbc(run_ws: str, catchment_id: int) -> float:
    """
    EXACTLY like OAT:
    Read last-time-step SGD (m3/d) from GHB flows in the CBC.
    MF6 convention: positive = INTO model; SGD (fresh GW -> sea) = sum of NEGATIVE flows, sign flipped.
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

def run_with_retry(catchment, filepaths, mf6, x5, year, coastal_buffer=200.0, max_retries=2):
    """
    x5 MUST be a list of 5 multipliers in order:
    [soilK, rockK, riv, ghb, rch]
    """
    soilK, rockK, riv, ghb, rch = x5
    attempts = []
    attempts.append((soilK, rockK, riv, ghb, rch, "baseline"))
    attempts.append((soilK, rockK, riv*0.90, ghb, rch, "riv*0.90"))
    attempts.append((soilK, rockK, riv, ghb*0.90, rch, "ghb*0.90"))

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
        except RuntimeError as e:
            if tried >= 1 + max_retries:
                raise
            print(f"  -> retry due to: {e}")

# ------------------- main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catchment", "-c", type=int, required=True)
    ap.add_argument("--year", "-y", type=int, required=True, help="Year for recharge/wells/sea level")
    ap.add_argument("--mf6", required=True, help="Path to mf6 executable")
    ap.add_argument("--nsamples", type=int, default=64,
                    help="Base Sobol sample size (total runs ≈ (k+2)*N when 2nd-order off). Prefer powers of 2.")
    ap.add_argument("--use-params", default="soilK,rockK,rch",
                    help="Comma-separated subset from {soilK,rockK,riv,ghb,rch}. Example: soilK,rockK,rch")
    ap.add_argument("--tight-bounds", action="store_true",
                    help="Use tighter bounds around calibration for faster, more local analysis")
    ap.add_argument("--sgd", choices=["ghb", "none"], default="ghb",
                    help="Compute SGD index from GHB (like OAT), or skip to save time")
    ap.add_argument("--no-clean", action="store_true",
                    help="Do NOT delete workspace between runs (faster I/O; use only if MF6 overwrites cleanly)")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for Sobol sampler (if supported)")
    ap.add_argument("--outdir", default=None,
                    help="Output folder for CSV/indices (default: <output-dir>/sensitivity_analysis)")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Path to input data folder (default: <project>/data/input)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Path to output folder (default: <project>/data/output)")
    args = ap.parse_args()

    INPUT  = pathlib.Path(args.data_root)  if args.data_root  else _DEFAULT_INPUT
    OUTPUT = pathlib.Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT
    if args.outdir is None:
        args.outdir = str(OUTPUT / "sensitivity_analysis")

    # Validate / parse parameter list
    all_names = ["soilK","rockK","riv","ghb","rch"]
    names = [p.strip() for p in args.use_params.split(",") if p.strip()]
    for n in names:
        if n not in all_names:
            raise ValueError(f"Unknown parameter '{n}'. Choose from {all_names}.")
    k = len(names)
    if k == 0:
        raise ValueError("You must select at least one parameter via --use-params.")

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
    filepaths = {k2: str(v) for k2, v in filepaths.items()}

    ws = run_workspace(filepaths, args.catchment)

    # --- Baseline run to get grid, mask, observed heads ---
    if not args.no_clean:
        safe_rmtree(ws)

    print("[baseline] building grid & mask …")
    sim0, dem_tr, dem_crs, catch_poly = run_with_retry(
        catchment=args.catchment, filepaths=filepaths, mf6=args.mf6,
        x5=[1,1,1,1,1], year=args.year, max_retries=2
    )

    obs = load_or_interpolate_obs_heads(
        well_path=filepaths["wells"], dem_path=filepaths["dem"],
        catch_poly=catch_poly, year=args.year, dem_tr=dem_tr, dem_crs=dem_crs,
        model_shape=sim0.shape,
        cache_dir=os.path.join(filepaths["output"], "cache", str(args.catchment), str(args.year))
    )
    mask = polygon_mask(catch_poly, sim0.shape, dem_tr)

    # --- Bounds ---
    if args.tight_bounds:
        bounds_map = {
            "soilK": [0.8, 1.6],
            "rockK": [1.2, 3.2],
            "riv"  : [0.8, 1.3],
            "ghb"  : [0.7, 1.1],
            "rch"  : [0.85, 1.15],
        }
    else:
        bounds_map = {
            "soilK": [0.5, 2.5],
            "rockK": [0.5, 3.0],
            "riv"  : [0.4, 1.5],
            "ghb"  : [0.4, 1.5],
            "rch"  : [0.8, 1.2],
        }

    problem = {
        "num_vars": k,
        "names": names,
        "bounds": [bounds_map[n] for n in names],
    }

    # --- Sampling ---
    print(f"[sobol] sampler={sobol_sampler_name}, generating samples …")
    X = sobol_sample(problem, N=args.nsamples, calc_second_order=False, seed=args.seed if sobol_sampler_name=="sobol" else None)
    n_runs = X.shape[0]
    print(f"[sobol] total MF6 runs: {n_runs}  (formula: (k+2)*N = ({k}+2)*{args.nsamples})")

    rmse_vals = np.full(n_runs, np.nan, dtype=float)
    sgd_vals  = np.full(n_runs, np.nan, dtype=float) if args.sgd == "ghb" else None

    # --- Run loop ---
    t0 = time.time()
    for i, row in enumerate(X, start=1):
        # start from all-ones and override only the active params
        pvals = {"soilK":1.0, "rockK":1.0, "riv":1.0, "ghb":1.0, "rch":1.0}
        for n, v in zip(names, row):
            pvals[n] = float(v)

        if not args.no_clean:
            safe_rmtree(ws)

        try:
            sim, *_ = run_with_retry(
                catchment=args.catchment, filepaths=filepaths, mf6=args.mf6, year=args.year,
                x5=[pvals["soilK"], pvals["rockK"], pvals["riv"], pvals["ghb"], pvals["rch"]],
                max_retries=2
            )
            rmse_vals[i-1] = rmse_vs_obs(sim, obs, mask)
            if sgd_vals is not None:
                sgd_vals[i-1] = read_sgd_m3d_from_cbc(ws, args.catchment)

        except Exception as e:
            print(f"[{i}/{n_runs}] run failed: {e}")

        if i % max(1, n_runs // 10) == 0:
            elapsed = time.time() - t0
            print(f"[sobol] {i}/{n_runs} ({100*i/n_runs:.1f}%)  elapsed {int(elapsed//60)}m{int(elapsed%60)}s")

    # --- Write raw results ---
    out_dir = pathlib.Path(args.outdir) / f"sobol_{args.catchment}_{args.year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"samples_outputs_c{args.catchment}_y{args.year}.csv"

    header = names + ["RMSE"]
    rows_data = np.column_stack([X, rmse_vals])
    if sgd_vals is not None:
        header.append("SGD_m3d")
        rows_data = np.column_stack([rows_data, sgd_vals])

    np.savetxt(out_csv, rows_data, delimiter=",", header=",".join(header), comments="")
    print(f"[out] wrote samples + outputs: {out_csv}")

    # --- Sobol indices (RMSE) ---
    print("finite RMSE =", np.isfinite(rmse_vals).sum(), "of", len(rmse_vals))
    if not np.isfinite(rmse_vals).all():
        print("[error] Some runs failed (NaN RMSE). Sobol requires a complete output vector.")
        print("        Re-run (or increase retries / loosen bounds) until all runs are finite.")
        return

    Sr = sobol.analyze(problem, rmse_vals, calc_second_order=False, print_to_console=False)
    print("\nSobol indices for RMSE:")
    for name, s1, s1c, st, stc in zip(problem["names"], Sr["S1"], Sr["S1_conf"], Sr["ST"], Sr["ST_conf"]):
        print(f"  {name:5s}  S1={s1: .3f} ±{s1c:.3f}   ST={st: .3f} ±{stc:.3f}")

    indices = {"RMSE": {"S1": Sr["S1"].tolist(), "ST": Sr["ST"].tolist(),
                        "S1_conf": Sr["S1_conf"].tolist(), "ST_conf": Sr["ST_conf"].tolist(),
                        "names": problem["names"]}}

    # --- Sobol indices (SGD) if requested ---
    if sgd_vals is not None:
        print("finite SGD =", np.isfinite(sgd_vals).sum(), "of", len(sgd_vals))
        if np.isfinite(sgd_vals).all():
            Ss = sobol.analyze(problem, sgd_vals, calc_second_order=False, print_to_console=False)
            print("\nSobol indices for SGD (m3/d):")
            for name, s1, s1c, st, stc in zip(problem["names"], Ss["S1"], Ss["S1_conf"], Ss["ST"], Ss["ST_conf"]):
                print(f"  {name:5s}  S1={s1: .3f} ±{s1c:.3f}   ST={st: .3f} ±{stc:.3f}")
            indices["SGD"] = {"S1": Ss["S1"].tolist(), "ST": Ss["ST"].tolist(),
                              "S1_conf": Ss["S1_conf"].tolist(), "ST_conf": Ss["ST_conf"].tolist(),
                              "names": problem["names"]}
        else:
            print("\n[warn] Some SGD values are NaN, skipping SGD Sobol analysis.")

    # Save indices JSON
    out_json = out_dir / f"sobol_indices_c{args.catchment}_y{args.year}.json"
    out_json.write_text(json.dumps(indices, indent=2))
    print(f"[out] wrote Sobol indices: {out_json}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
