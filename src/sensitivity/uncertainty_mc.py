#!/usr/bin/env python3
# uncertainty_mc.py
# Monte-Carlo uncertainty for user-selected multipliers (epistemic uncertainty).
# - Reuses your MF6 setup/run helper (setup_and_run_modflow)
# - RMSE vs observed heads; optional SGD from GHB (like OAT/Sobol)
# - Distributions: uniform within bounds, or truncated normal around a center value
# - Writes samples+outputs CSV and a text summary; optional save of per-run heads and percentiles

import os, sys, json, argparse, pathlib, time, shutil, math
import numpy as np

# Keep GDAL/RasterIO memory tame
os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("RASTERIO_MAXIMUM_RAM", "512MB")

# raster
import rasterio
from rasterio.features import rasterize

# MF helpers from your project
from core.modflow_setup import setup_and_run_modflow, load_or_interpolate_obs_heads

ROOT   = pathlib.Path(__file__).resolve().parents[2]
DATA   = ROOT / "data"
INPUT  = DATA / "input"
OUTPUT = DATA / "output"


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
    EXACTLY like OAT/Sobol:
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
    x5 MUST be a list of 5 multipliers in order: [soilK, rockK, riv, ghb, rch]
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


def truncated_normal(mean, sd, low, high, size, rng):
    """Draw from N(mean, sd^2) truncated to [low, high]."""
    # simple rejection sampler (fine for modest sizes)
    out = np.empty(size, dtype=float)
    k = 0
    while k < size:
        draws = rng.normal(mean, sd, size=(size - k))
        keep = (draws >= low) & (draws <= high)
        n_keep = keep.sum()
        if n_keep > 0:
            out[k:k+n_keep] = draws[keep][:n_keep]
            k += n_keep
    return out


# ------------------- main -------------------
def main():
    ap = argparse.ArgumentParser(description="Monte-Carlo uncertainty for MF6 multipliers")
    ap.add_argument("--catchment", "-c", type=int, required=True)
    ap.add_argument("--year", "-y", type=int, required=True, help="Year for recharge/wells/sea level")
    ap.add_argument("--mf6", required=True, help="Path to mf6 executable")

    ap.add_argument("--use-params", default="soilK,rockK,rch",
                    help="Comma-separated subset from {soilK,rockK,riv,ghb,rch}. Example: soilK,rockK,rch")
    ap.add_argument("--tight-bounds", action="store_true",
                    help="Use tighter bounds around calibration (recommended)")
    ap.add_argument("--dist", choices=["uniform", "truncnorm"], default="uniform",
                    help="Sampling distribution within bounds (uniform) or around centers (truncnorm)")
    ap.add_argument("--center", default="soilK=1,rockK=2.2,riv=1,ghb=0.9,rch=1.0",
                    help="Comma list of param=center used for truncnorm (ignored for uniform)")
    ap.add_argument("--sigma-frac", default="soilK=0.15,rockK=0.20,riv=0.10,ghb=0.08,rch=0.08",
                    help="Comma list of param=relative_sigma (e.g., 0.15 means 15% of center). truncnorm only.")
    ap.add_argument("--n", type=int, default=300, help="Number of Monte-Carlo draws")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")

    ap.add_argument("--sgd", choices=["ghb", "none"], default="ghb",
                    help="Compute SGD index from GHB (like OAT), or skip to save time")
    ap.add_argument("--no-clean", action="store_true",
                    help="Do NOT delete workspace between runs (faster I/O; use only if MF6 overwrites cleanly)")
    ap.add_argument("--save-heads", action="store_true",
                    help="Save each run's head raster for later percentile maps (disk heavy)")
    ap.add_argument("--save-percentiles", default="",
                    help="Comma-separated percentile list to compute across saved heads, e.g. 5,50,95 (requires --save-heads)")

    ap.add_argument("--outdir", default=str(OUTPUT / "uncertainty_mc"),
                    help="Output folder for CSV/summary/optional heads")
    args = ap.parse_args()

    all_names = ["soilK","rockK","riv","ghb","rch"]
    names = [p.strip() for p in args.use_params.split(",") if p.strip()]
    for n in names:
        if n not in all_names:
            raise ValueError(f"Unknown parameter '{n}'. Choose from {all_names}.")
    if len(names) == 0:
        raise ValueError("You must select at least one parameter via --use-params.")

    # Filepaths (use yearly recharge tif)
    filepaths = {
        "dem"         : INPUT / "dem" / "elevation_sweden.tif",
        "catchment"   : INPUT / "shapefiles" / "catchment" / "bsdbs.shp",
        "recharge"    : DATA  / "output" / "recharge_yearly" / f"recharge_egdi_gldas_{args.year}.tif",
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
    filepaths = {k: str(v) for k, v in filepaths.items()}
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

    # --- Bounds (same as your Sobol script) ---
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

    # --- Centers and sigmas for truncnorm ---
    def parse_map(s, default):
        out = dict(default)
        if not s:
            return out
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                k = k.strip()
                v = float(v.strip())
                if k in out:
                    out[k] = v
        return out

    default_center = {"soilK":1.0, "rockK":2.2, "riv":1.0, "ghb":0.9, "rch":1.0}
    default_sigmaf = {"soilK":0.15,"rockK":0.20,"riv":0.10,"ghb":0.08,"rch":0.08}
    centers = parse_map(args.center, default_center)
    sigmaf  = parse_map(args.sigma_frac, default_sigmaf)

    # --- Sampling ---
    rng = np.random.default_rng(args.seed)

    def draw_param(name, size):
        lo, hi = bounds_map[name]
        if args.dist == "uniform":
            return rng.uniform(lo, hi, size=size)
        else:
            mu = centers[name]
            sd = sigmaf[name] * abs(mu)
            return truncated_normal(mu, sd, lo, hi, size, rng)

    X = np.zeros((args.n, len(names)), dtype=float)
    for j, nm in enumerate(names):
        X[:, j] = draw_param(nm, size=args.n)

    # --- Run loop ---
    out_dir = pathlib.Path(args.outdir) / f"mc_{args.catchment}_{args.year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rmse_vals = np.full(args.n, np.nan, dtype=float)
    sgd_vals  = np.full(args.n, np.nan, dtype=float) if args.sgd == "ghb" else None

    # optional per-run heads
    heads_dir = out_dir / "heads_runs"
    if args.save_heads:
        heads_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, row in enumerate(X, start=1):
        # start from ones, override active params
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

            if args.save_heads:
                # save run head as a GeoTIFF matching baseline transform/CRS
                fp = heads_dir / f"head_run_{i:04d}.tif"
                with rasterio.open(
                    fp, "w",
                    driver="GTiff",
                    height=sim.shape[0], width=sim.shape[1],
                    count=1, dtype="float32",
                    crs=dem_crs, transform=dem_tr,
                    nodata=np.nan, compress="LZW"
                ) as dst:
                    dst.write(sim.astype("float32"), 1)

        except Exception as e:
            print(f"[{i}/{args.n}] run failed: {e}")

        if i % max(1, args.n // 10) == 0:
            elapsed = time.time() - t0
            print(f"[mc] {i}/{args.n} ({100*i/args.n:.1f}%)  elapsed {int(elapsed//60)}m{int(elapsed%60)}s")

    # --- Write CSV ---
    header = names + ["RMSE"]
    data = np.column_stack([X, rmse_vals])
    if sgd_vals is not None:
        header.append("SGD_m3d")
        data = np.column_stack([data, sgd_vals])

    out_csv = out_dir / f"mc_samples_outputs_c{args.catchment}_y{args.year}.csv"
    np.savetxt(out_csv, data, delimiter=",", header=",".join(header), comments="")
    print(f"[out] wrote samples + outputs: {out_csv}")

    # --- Summary text (medians and 95% intervals) ---
    def summarize(arr, name):
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return f"{name}: no finite values"
        p50 = np.percentile(a, 50)
        p05 = np.percentile(a, 5)
        p95 = np.percentile(a, 95)
        return f"{name}: median={p50:.3f}  [P5={p05:.3f}, P95={p95:.3f}] (n={a.size})"

    summary_lines = []
    summary_lines.append(summarize(rmse_vals, "RMSE (m)"))
    if sgd_vals is not None:
        summary_lines.append(summarize(sgd_vals, "SGD (m3/d)"))

    out_txt = out_dir / f"summary_c{args.catchment}_y{args.year}.txt"
    out_txt.write_text("\n".join(summary_lines))
    print("[summary]\n" + "\n".join(summary_lines))

    # --- Optional percentile maps (requires --save-heads) ---
    if args.save_heads and args.save_percentiles:
        pct_list = [int(p.strip()) for p in args.save_percentiles.split(",") if p.strip()]
        if pct_list:
            print(f"[maps] computing percentiles {pct_list} across saved heads …")
            # Load all heads into memory (simple approach; OK for moderate grid size and N)
            files = sorted(heads_dir.glob("head_run_*.tif"))
            if files:
                arrs = []
                for fp in files:
                    with rasterio.open(fp) as src:
                        arrs.append(src.read(1).astype("float32"))
                stack = np.stack(arrs, axis=0)  # shape: (N, rows, cols)
                for p in pct_list:
                    q = np.nanpercentile(stack, p, axis=0)
                    outp = out_dir / f"head_P{p}.tif"
                    with rasterio.open(
                        outp, "w",
                        driver="GTiff",
                        height=q.shape[0], width=q.shape[1],
                        count=1, dtype="float32",
                        crs=dem_crs, transform=dem_tr,
                        nodata=np.nan, compress="LZW"
                    ) as dst:
                        dst.write(q.astype("float32"), 1)
                    print(f"[maps] wrote {outp}")
            else:
                print("[maps] no saved heads found; skipped.")

    print(f"[done] Uncertainty MC written to: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
