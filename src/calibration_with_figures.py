
#!/usr/bin/env python3
# calibration_with_figures.py
# Calibration wrapper for SGD model + end-of-run figures
# - Observed heads processed identically to initial heads
# - Initial-head interpolation cached via runtime patch (no changes to other modules)
# - Memoized objective (no key collapse), coarse probe, bigger eps for optimizer
# 2025-09-30 Arya Vijayan


import argparse
import os
import hashlib
import numpy as np
import pandas as pd
import json
import os as _os
_os.environ["MPLBACKEND"] = "Agg"      
import matplotlib
matplotlib.use("Agg", force=True)
print("Matplotlib backend ->", matplotlib.get_backend())
import matplotlib.pyplot as plt

import sys
from scipy.optimize import minimize
import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import csv
from rasterio.features import rasterize

# Your modules
from modflow_setup import setup_and_run_modflow, load_or_interpolate_obs_heads
import sgd_utils
import modflow_setup as mfs
from sgd_utils import interpolate_well_heads
from matplotlib.ticker import ScalarFormatter



# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _list_float(s):
    if s is None:
        return None
    s = s.strip()
    if s in ("", "''", '""'):
        return None
    return [float(x) for x in s.split(",") if x.strip()]

def _dict_float(s):
    if s is None:
        return None
    s = s.strip()
    if s in ("", "''", '""'):
        return None
    out = {}
    for tok in s.split(","):
        k, v = tok.split(":")
        out[int(k.strip())] = float(v.strip())
    return out

# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate SGD model parameters")
    parser.add_argument('-c','--catchment', type=int, required=True,
                        help="Catchment ID (ID_BSDB)")
    parser.add_argument('-y','--year', type=int, required=True,
                        help="Year for recharge and well heads calibration")
    parser.add_argument('--mf6', required=True, help="Path to MF6 executable")

    # Optional toggles
    parser.add_argument('--maxiter', type=int, default=60, help="Max optimizer iterations (default 60)")
    parser.add_argument('--ftol', type=float, default=1e-3, help="Function tolerance (default 1e-3)")
    parser.add_argument('--fix-riv', type=float, default=None, help="Fix river cond multiplier")
    parser.add_argument('--fix-ghb', type=float, default=None, help="Fix GHB cond multiplier")
    parser.add_argument('--no-figures', action='store_true', help="Skip saving figures (faster)")
    parser.add_argument('--fix-soil', type=float, default=None, help="Fix soilK multiplier")
    parser.add_argument('--fix-rock', type=float, default=None, help="Fix rockK multiplier")
    parser.add_argument('--skip-probe', action='store_true', help="Skip coarse probe (single-run/validation)")
    parser.add_argument('--fix-rch', type=float, default=None, help="Fix recharge multiplier")

    # Zoning knobs
    parser.add_argument("--rch-elev-bins",     type=_list_float, default=None,
                        help="Comma-separated elevation edges (m), e.g. 0,10,30,60,200")
    parser.add_argument("--rch-elev-factors",  type=_list_float, default=None,
                        help="Recharge multipliers per elevation band, e.g. 1.0,1.1,1.25,1.35")
    parser.add_argument("--rch-soil-factors",  type=_dict_float, default=None,
                        help="Recharge multipliers by soil class, e.g. 1:1.00,2:1.10,3:1.05")
    parser.add_argument("--k-soil-factors",    type=_dict_float, default=None,
                        help="Soil K multipliers by soil class, e.g. 1:1.0,2:0.7,3:0.9")
    
    parser.add_argument('--no-grid-fallback', action='store_true',
                    help='Disable the expensive grid-search fallback')


    args = parser.parse_args()

    # Validate paired elevation inputs if both provided
    if args.rch_elev_bins and args.rch_elev_factors:
        nbands = len(args.rch_elev_bins) - 1
        if nbands != len(args.rch_elev_factors):
            parser.error(
                f"--rch-elev-bins defines {nbands} bands "
                f"but --rch-elev-factors has {len(args.rch_elev_factors)} values"
            )

    return args




def print_iter(xk):
    xk = np.asarray(xk, dtype=float)
    print(f"  ↪ accepted params = [{', '.join(f'{v:.3f}' for v in xk)}]")


# ──────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ──────────────────────────────────────────────────────────────────────────────
def plot_head_maps(obs, sim, catch_id, year, output_dir, catch_mask):
    """
    Save 3 side-by-side maps:
      1) Interpolated Observed Heads
      2) Simulated Heads
      3) Residuals (Sim − Obs)

    - Applies a catchment mask so nothing outside the basin is shown
    - Cleans nodata/extreme values
    - Uses robust color limits (2nd–98th percentiles)
    - Makes NaNs transparent
    - Uses a symmetric diverging scale for residuals
    """
    import matplotlib as mpl
    os.makedirs(output_dir, exist_ok=True)

    # --- copy & clean ---
    obs_plot = np.asarray(obs, dtype=float).copy()
    sim_plot = np.asarray(sim, dtype=float).copy()

    # common cleaning: non-finite → NaN, clip absurd outliers
    for arr in (obs_plot, sim_plot):
        arr[~np.isfinite(arr)] = np.nan
        arr[(arr < -1000) | (arr > 5000)] = np.nan  # adjust if your heads can be outside this

    # apply basin mask (hide everything outside)
    if catch_mask is not None:
        obs_plot = np.where(catch_mask, obs_plot, np.nan)
        sim_plot = np.where(catch_mask, sim_plot, np.nan)

    # diagnostics to console
    def _rng(a):
        if np.isfinite(a).any():
            return float(np.nanmin(a)), float(np.nanmax(a)), int(np.isfinite(a).sum())
        return np.nan, np.nan, 0
    omin, omax, on = _rng(obs_plot)
    smin, smax, sn = _rng(sim_plot)
    print(f"[plot] obs valid={on} min/max={omin:.2f}…{omax:.2f} | sim valid={sn} min/max={smin:.2f}…{smax:.2f}")

    # residuals
    residuals = np.where(catch_mask, sim_plot - obs_plot, np.nan)

    # --- robust color limits for heads ---
    valid_heads = np.concatenate([
        obs_plot[np.isfinite(obs_plot)],
        sim_plot[np.isfinite(sim_plot)]
    ]) if (np.isfinite(obs_plot).any() or np.isfinite(sim_plot).any()) else np.array([])

    if valid_heads.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.nanpercentile(valid_heads, [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(np.nanmin(valid_heads)), float(np.nanmax(valid_heads))

    # symmetric limits for residuals
    if np.isfinite(residuals).any():
        rlim = float(np.nanmax(np.abs(residuals)))
        if not np.isfinite(rlim) or rlim == 0:
            rlim = 1.0
    else:
        rlim = 1.0

    # --- colormaps with transparent NaNs ---
    cmap = mpl.cm.viridis.copy()
    cmap.set_bad(alpha=0.0)
    cmap_div = mpl.cm.RdBu_r.copy()
    cmap_div.set_bad(alpha=0.0)
    
    # --- plot ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axs[0].imshow(obs_plot, origin='upper', vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].set_title("Interpolated Observed Heads (m)")
    axs[0].axis('off')
    cbar0 = plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    cbar0.ax.set_ylabel("m", rotation=90, va="center")
    cbar0.formatter = ScalarFormatter(useOffset=False); cbar0.update_ticks()

    im1 = axs[1].imshow(sim_plot, origin='upper', vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1].set_title("Simulated Heads (m)")
    axs[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    cbar1.ax.set_ylabel("m", rotation=90, va="center")
    cbar1.formatter = ScalarFormatter(useOffset=False); cbar1.update_ticks()

    im2 = axs[2].imshow(residuals, origin='upper', vmin=-rlim, vmax=rlim, cmap=cmap_div)
    axs[2].set_title("Residuals (Sim − Obs) (m)")
    axs[2].axis('off')
    cbar2 = plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    cbar2.ax.set_ylabel("m", rotation=90, va="center")
    cbar2.formatter = ScalarFormatter(useOffset=False); cbar2.update_ticks()

    plt.suptitle(f"Catchment {catch_id} — Year {year}", y=0.98)
    plt.tight_layout()

    outpath = os.path.join(output_dir, f"head_maps_c{catch_id}_y{year}.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"✓ Saved head maps: {outpath}")


def plot_rmse_convergence(rmse_history, catch_id, year, output_dir):
    import os, numpy as np, matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    os.makedirs(output_dir, exist_ok=True)
    vals = np.asarray(rmse_history, dtype=float)

    # Hide penalties entirely in the plot
    is_good = np.isfinite(vals) & (vals < 1e5)
    if not np.any(is_good):
        return
    vals_plot = vals.copy()
    vals_plot[~is_good] = np.nan

    ymax = max(1.2 * np.nanmax(vals_plot), 1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(vals_plot, marker='o', linewidth=1)

    ax.set_title(f"Optimization Convergence (Catchment {catch_id}, {year})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE (m)")
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.4)

    n_pen = int(np.sum(~is_good & np.isfinite(vals)))
    if n_pen > 0:
        ax.text(0.98, 0.02, f"{n_pen} penalty evals hidden",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9, alpha=0.7)

    outpath = os.path.join(output_dir, f"rmse_convergence_c{catch_id}_y{year}.png")
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved RMSE convergence plot: {outpath}")



'''
def plot_rmse_convergence(rmse_history, catch_id, year, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(rmse_history, marker='o')
    plt.title(f"Optimization Convergence (Catchment {catch_id}, {year})")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE (m)")
    plt.grid(True, alpha=0.4)
    outpath = os.path.join(output_dir, f"rmse_convergence_c{catch_id}_y{year}.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved RMSE convergence plot: {outpath}")

'''
def plot_wells_map(wells, catch_poly, crs, catch_id, year, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if wells is None or getattr(wells, "empty", True):
        print("⚠ No wells to plot."); return
    if wells.crs != crs:
        wells = wells.to_crs(crs)
    catch_gs = gpd.GeoSeries([catch_poly], crs=crs)
    ax = catch_gs.plot(edgecolor='black', facecolor='none', figsize=(7,7))
    wells.plot(ax=ax, markersize=8)
    ax.set_title(f"Observation Wells Used — Catchment {catch_id} ({year})")
    ax.set_axis_off()
    outpath = os.path.join(output_dir, f"wells_map_c{catch_id}_y{year}.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved wells map: {outpath}")

    
def print_global_budget_from_lst(lst_path):
    """
    Parse the last GLOBAL BUDGET table from mfsim.lst and print % discrepancy.
    """
    if not os.path.exists(lst_path):
        print(f"[budget] list file not found: {lst_path}")
        return
    in_total = out_total = None
    with open(lst_path, "r", errors="ignore") as f:
        lines = f.readlines()
    # scan from bottom for the 'GLOBAL BUDGET' block
    for i in range(len(lines)-1, -1, -1):
        if "GLOBAL BUDGET" in lines[i]:
            for j in range(i, min(i+200, len(lines))):
                if "TOTAL IN " in lines[j]:
                    in_total = float(lines[j].split()[-1])
                if "TOTAL OUT" in lines[j]:
                    out_total = float(lines[j].split()[-1])
                if (in_total is not None) and (out_total is not None):
                    break
            break
    if (in_total is None) or (out_total is None):
        print("[budget] could not find totals in list file.")
        return
    diff = abs(in_total - out_total)
    mean_throughflow = 0.5 * (in_total + out_total)
    pct = 100.0 * diff / mean_throughflow if mean_throughflow > 0 else np.nan
    print(f"[budget] TOTAL IN={in_total:.3e}, OUT={out_total:.3e} | diff={diff:.3e} ({pct:.2f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers for obs heads & plotting
# ──────────────────────────────────────────────────────────────────────────────
def _dem_on_model_grid(dem_path, dem_tr, dem_crs, shape):
    nrows, ncols = shape
    with rasterio.open(dem_path) as src:
        with WarpedVRT(src, crs=dem_crs, transform=dem_tr,
                       width=ncols, height=nrows,
                       resampling=Resampling.bilinear) as vrt:
            dem = vrt.read(1).astype(float)
            if vrt.nodata is not None:
                dem[dem == vrt.nodata] = np.nan
    return dem


def _wells_filtered_for_plot(well_path, catch_poly, year, dem_crs):
    """Filter wells for plotting consistent with interpolation pipeline."""
    # Try default layer first (like sgd_utils), then explicit layer
    try:
        wells = gpd.read_file(well_path)
        if wells.empty:
            raise ValueError("default layer empty")
    except Exception:
        wells = gpd.read_file(well_path, layer="brunnar")

    # Parse date
    if 'nivadatum' in wells.columns:
        wells['date'] = pd.to_datetime(wells['nivadatum'].astype(str), format='%Y%m%d', errors='coerce')
    elif 'borrdatum' in wells.columns:
        wells['date'] = pd.to_datetime(wells['borrdatum'].astype(str), format='%Y%m%d', errors='coerce')
    else:
        wells['date'] = pd.NaT

    # Year filter
    if year is not None and 'date' in wells.columns:
        wells = wells[wells['date'].dt.year == year].copy()

    # Reproject, clip & clean geometries
    if wells.crs != dem_crs:
        wells = wells.to_crs(dem_crs)
    wells = wells[wells.geometry.notnull()].copy()
    wells = wells[wells.geometry.intersects(catch_poly)].copy()
    wells.geometry = wells.geometry.apply(
        lambda g: g.geoms[0] if getattr(g, "geom_type", None) == "MultiPoint" else g
    )
    wells = wells[wells.geometry.geom_type == 'Point'].copy()
    return wells


def build_observed_heads_identical(filepaths, catch_poly, year, dem_tr, dem_crs, model_shape):
    """Observed heads using EXACT same pipeline as initial heads."""
    dem_grid = _dem_on_model_grid(filepaths['dem'], dem_tr, dem_crs, model_shape)
    obs_grid = interpolate_well_heads(
        well_gpkg        = filepaths['wells'],
        catchment_geom   = [catch_poly],
        target_shape     = model_shape,
        target_transform = dem_tr,
        target_crs       = dem_crs,
        dem_array        = dem_grid,
        year             = year
    )
    wells_for_plot = _wells_filtered_for_plot(filepaths['wells'], catch_poly, year, dem_crs)
    return obs_grid, wells_for_plot

def make_well_support_mask(wells_gdf, dem_tr, shape, radius_m=3000):
    """
    Cells within radius_m of any well; returns boolean array with 'shape'.
    If no wells, fallback to all-True.
    """
    if wells_gdf is None or wells_gdf.empty:
        return np.ones(shape, dtype=bool)
    ring = wells_gdf.buffer(radius_m)
    mask = rasterize([(g, 1) for g in ring.geometry],
                     out_shape=shape, transform=dem_tr,
                     fill=0, all_touched=True, dtype="uint8").astype(bool)
    return mask

# ──────────────────────────────────────────────────────────────────────────────
# Initial-head caching (runtime patch; leaves other modules unmodified on disk)
# ──────────────────────────────────────────────────────────────────────────────
def _transform_fingerprint(tr, shape, crs):
    payload = (
        float(tr.a), float(tr.b), float(tr.c),
        float(tr.d), float(tr.e), float(tr.f),
        int(shape[0]), int(shape[1]),
        str(crs)
    )
    return hashlib.sha1("|".join(map(str, payload)).encode("utf-8")).hexdigest()[:12]


def install_initial_head_cache(catch_id, year, dem_tr, dem_crs, model_shape,
                               cache_root="data/output/cache"):
    cache_dir = os.path.join(cache_root, str(catch_id))
    os.makedirs(cache_dir, exist_ok=True)

    fp = _transform_fingerprint(dem_tr, model_shape, dem_crs)
    cache_path = os.path.join(cache_dir, f"initial_head_cache_y{year}_{fp}.npz")

    original = sgd_utils.interpolate_well_heads
    _mem = {}

    def cached_interpolate(well_gpkg, catchment_geom, target_shape, target_transform, target_crs, dem_array, year=None):
        key = (fp, year)
        if key in _mem:
            return _mem[key].copy()
        if os.path.exists(cache_path):
            try:
                arr = np.load(cache_path)["h0"]
                _mem[key] = arr
                print(f"[cache] Reusing initial heads from {cache_path}")
                return arr.copy()
            except Exception as e:
                print(f"[cache] Failed to read {cache_path} ({e}); recomputing…")
        arr = original(well_gpkg, catchment_geom, target_shape, target_transform, target_crs, dem_array, year=year)
        try:
            np.savez_compressed(cache_path, h0=arr.astype(np.float32))
            print(f"[cache] Saved initial heads to {cache_path}")
        except Exception as e:
            print(f"[cache] Could not write {cache_path} ({e})")
        _mem[key] = arr
        return arr.copy()

    # Patch both the module function and the name imported in modflow_setup
    sgd_utils.interpolate_well_heads = cached_interpolate
    mfs.interpolate_well_heads = cached_interpolate
    sys.modules['modflow_setup'].__dict__['interpolate_well_heads'] = cached_interpolate


# ──────────────────────────────────────────────────────────────────────────────
# Objective with robust memoization (no key rounding) + early stop
# ──────────────────────────────────────────────────────────────────────────────

zone_kwargs = {}
    
def make_objective(catch_id, year, mf6_exe, filepaths, obs_grid, catch_mask,
                   stop_patience=999999, stop_tol=0.0, max_evals=None):
    """
    Returns an objective function with:
      - memoization to avoid repeated MF runs at the *same* x,
      - best-simulated field retained to avoid an extra final run.
    """
    cache = {}  # key: exact tuple(x) -> (rmse, sim_heads)
    best = {'rmse': np.inf, 'x': None, 'sim': None}
    history = []      # RMSE per call
    x_history = []    # params per call

    def key_exact(x):
        return tuple(np.asarray(x, dtype=float).tolist())

    def objective_inner(x):
        x = np.asarray(x, dtype=float)
        if (max_evals is not None) and (len(history) >= max_evals):
            raise SystemExit("Reached max evaluations")

        k = key_exact(x)
        print(f"[iter] eval #{len(history)+1}")

        if k in cache:
            rmse, sim_heads = cache[k]
            print(f"[eval] x={x}  RMSE={rmse:.4f}  (cached)")
        else:
            try:
                ret = setup_and_run_modflow(
                    catch_id, filepaths,
                    coastal_buffer=200.0,
                    mf6_exe=mf6_exe,
                    soilK_multiplier=x[0],
                    rockK_multiplier=x[1],
                    riv_cond_multiplier=x[2] if len(x) > 2 else 1.0,
                    ghb_cond_multiplier=x[3] if len(x) > 3 else 1.0,
                    recharge_year=year,
                    rch_multiplier=x[4] if len(x) > 4 else 1.0,
                    **zone_kwargs
                )
                sim_heads = ret[0] if isinstance(ret, tuple) else ret
            except Exception as e:
                print(f"[objective] run failed: {e}")
                rmse, sim_heads = 1e6, None
                cache[k] = (rmse, sim_heads)
                history.append(rmse)
                x_history.append(x.tolist())
                return rmse

            # Clean & RMSE
            sim_clean = sim_heads.astype(float).copy()
            sim_clean[~np.isfinite(sim_clean)] = np.nan
            sim_clean[(sim_clean < -1000) | (sim_clean > 5000)] = np.nan

            obs_clean = obs_grid.astype(float).copy()
            obs_clean[~np.isfinite(obs_clean)] = np.nan

            mask = catch_mask & np.isfinite(sim_clean) & np.isfinite(obs_clean)
            print(f"[cmp] cells compared = {int(mask.sum())} ({mask.mean()*100:.1f}%)")
            try:
                smin, smax = np.nanmin(sim_clean), np.nanmax(sim_clean)
            except ValueError:
                smin = smax = np.nan
            try:
                omin, omax = np.nanmin(obs_clean), np.nanmax(obs_clean)
            except ValueError:
                omin = omax = np.nan
            print(f"[heads] sim min/max: {smin:.2f} … {smax:.2f} m | obs min/max: {omin:.2f} … {omax:.2f} m")

            rmse = np.inf if not np.any(mask) else np.sqrt(
                np.nanmean((sim_clean[mask] - obs_clean[mask])**2)
            )
            cache[k] = (rmse, sim_heads)
            print(f"[eval] x={x}  RMSE={rmse:.4f}")

        # record history
        history.append(rmse)
        x_history.append(x.tolist())

        # track best
        if rmse + 1e-12 < best['rmse']:
            best['rmse'] = rmse
            best['x'] = x.copy()
            best['sim'] = sim_heads

        return rmse

    return objective_inner, best, history, x_history




def run_fallback_grid_search(catch_id, year, mf6_exe, filepaths, obs_grid, catch_mask):
    soil_vals = [0.5, 0.8, 1.0, 1.2, 1.5]
    rock_vals = [0.5, 0.8, 1.0, 1.5, 2.0]
    riv_vals  = [0.5, 1.0, 1.5]
    ghb_vals  = [0.5, 1.0, 1.5]
    rch_vals  = [0.8, 1.0, 1.2]


    best = {'rmse': np.inf, 'x': None, 'sim': None}
    tried = 0

    for s in soil_vals:
        for r in rock_vals:
            for rv in riv_vals:
                for g in ghb_vals:
                    for rch in rch_vals:
                        tried += 1
                        try:
                            ret = setup_and_run_modflow(
                            catch_id, filepaths,
                            coastal_buffer=200.0,
                            mf6_exe=mf6_exe,
                            soilK_multiplier=s,
                            rockK_multiplier=r,
                            riv_cond_multiplier=rv,
                            ghb_cond_multiplier=g,
                            recharge_year=year,
                            rch_multiplier=rch,
                            **zone_kwargs
                            )
                            sim_heads = ret[0] if isinstance(ret, tuple) else ret

                            # RMSE here (inside try)
                            sim_clean = sim_heads.astype(float)
                            sim_clean[~np.isfinite(sim_clean)] = np.nan
                            sim_clean[(sim_clean < -1000) | (sim_clean > 5000)] = np.nan

                            obs_clean = obs_grid.astype(float)
                            obs_clean[~np.isfinite(obs_clean)] = np.nan

                            mask = catch_mask & np.isfinite(sim_clean) & np.isfinite(obs_clean)
                            rmse = np.inf if not np.any(mask) else np.sqrt(np.nanmean((sim_clean[mask] - obs_clean[mask])**2))

                            print(f"[grid] soil={s:.2f} rock={r:.2f} riv={rv:.2f} ghb={g:.2f} rch={rch:.2f}  →  RMSE={rmse:.4f}")

                            if rmse < best['rmse']:
                                best['rmse'] = rmse
                                best['x'] = np.array([s, r, rv, g, rch], dtype=float)
                                best['sim'] = sim_heads
                        except Exception as e:
                            print(f"[grid] run failed for s={s}, r={r}, rv={rv}, g={g}, rch={rch}: {e}")
                            continue
    print(f"[grid] tried {tried} combos, best RMSE={best['rmse']:.4f} at x={best['x']}")
    return best

                        
# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    catch_id, year, mf6_exe = args.catchment, args.year, args.mf6
    global zone_kwargs
    zone_kwargs = dict(             # <<< now you update the global used by make_objective
        rch_elev_bins    = args.rch_elev_bins,
        rch_elev_factors = args.rch_elev_factors,
        rch_soil_factors = args.rch_soil_factors,
        k_soil_factors   = args.k_soil_factors,
    )
      
    filepaths = {
        'dem':         'data/input/dem/elevation_sweden.tif',
        'catchment':   'data/input/shapefiles/catchment/bsdbs.shp',
        'recharge':    'data/input/recharge_data_selection_for_calibration.csv',
        'soil_perm':   'data/input/aquifer_data/genomslapplighet/genomslapplighet.gpkg',
        'soil_depth':  'data/input/aquifer_data/jorddjupsmodell/jorddjupsmodell_10x10m.tif',
        'conductivity':'data/input/other_rasters/hydraulic_conductivity.tif',
        'sea_level':   'data/input/sea_level/yearly_average_sea_level.csv',
        'coast':       'data/input/shapefiles/coast_line/coastline.shp',
        'wells':       'data/input/well_data/brunnar.gpkg',
        'rivers':      'data/input/shapefiles/surface_water/Surface_water/hl_riks.shp',
        'lakes':       'data/input/shapefiles/surface_water/scandinavian_waters_polygons.shp',
        'output':      'data/output'
        
    }
    os.makedirs(filepaths['output'], exist_ok=True)
    # --- USE YEAR-SPECIFIC RECHARGE TIFs (m/day) ---
    RECH_DIR = r"data/output/recharge_yearly"
    RECH_BASENAME = "recharge_egdi_gldas_{year}.tif"   # matches the writer script
    tif_path = os.path.join(RECH_DIR, RECH_BASENAME.format(year=year))
    if not os.path.exists(tif_path):
        raise FileNotFoundError(
            f"Recharge raster for year {year} not found:\n  {tif_path}\n"
            "Make sure you ran make_yearly_recharge_rasters.py and the --basename matches."
        )
    filepaths['recharge'] = tif_path
    recharge_abs = os.path.abspath(filepaths['recharge'])
    print(f"[recharge] USING FILE: {recharge_abs}")
    print(f"[recharge] Exists? {os.path.exists(recharge_abs)}  Size(bytes)={os.path.getsize(recharge_abs) if os.path.exists(recharge_abs) else 'NA'}")
    print(f"[recharge] Using per-year GeoTIFF (m/day): {filepaths['recharge']}")


    if str(filepaths['recharge']).lower().endswith(".csv"):
        try:
            rch = pd.read_csv(filepaths['recharge'], sep=';')  # semicolon for your file
            if 'year' in rch.columns:
                print(f"[recharge CSV] rows={len(rch)}, years=({rch['year'].min()}…{rch['year'].max()})")
                yr_rows = rch.loc[rch['year'] == year]
                print(f"[recharge CSV] year={year} rows={len(yr_rows)} "
                    f"Recharge_mm_year min/max="
                    f"{yr_rows['Recharge_mm_year'].min() if not yr_rows.empty else 'NA'}/"
                    f"{yr_rows['Recharge_mm_year'].max() if not yr_rows.empty else 'NA'}")
                if yr_rows.empty:
                    print(f"[recharge CSV][WARNING] No rows found for year={year}.")
            else:
                print("[recharge CSV][WARNING] No 'year' column — verify your schema/headers.")
        except Exception as e:
            print(f"[recharge CSV] could not read: {e}")

    
    print(f"▶ Starting calibration for catchment={catch_id}, year={year}")

    # 1) Baseline run to get grid/meta
    try:
        ret = setup_and_run_modflow(catch_id, filepaths,
            coastal_buffer=200.0,
            mf6_exe=mf6_exe,
            soilK_multiplier=1.0,
            rockK_multiplier=1.0,
            riv_cond_multiplier=1.0,
            ghb_cond_multiplier=1.0,
            recharge_year=year,
            rch_multiplier=1.0,
            **zone_kwargs
        )
        if isinstance(ret, tuple) and len(ret) == 5:
            head0, dem_tr, dem_crs, catch_poly, lst_path = ret
        else:
            head0, dem_tr, dem_crs, catch_poly = ret
            lst_path = None
        if lst_path:
            print_global_budget_from_lst(lst_path)


    except Exception as e:
        print("[baseline] Initial run failed, retrying with softer conductances…")
        # Use CLI fixed values if provided; otherwise gently reduce RIV/GHB for stability
        soil_fallback = float(args.fix_soil) if args.fix_soil is not None else 1.0
        rock_fallback = float(args.fix_rock) if args.fix_rock is not None else 1.0
        riv_fallback  = float(args.fix_riv)  if args.fix_riv  is not None else 0.85
        ghb_fallback  = float(args.fix_ghb)  if args.fix_ghb  is not None else 0.90

        ret = setup_and_run_modflow(
            catch_id, filepaths,
            coastal_buffer=200.0,
            mf6_exe=mf6_exe,
            soilK_multiplier=soil_fallback,
            rockK_multiplier=rock_fallback,
            riv_cond_multiplier=riv_fallback,
            ghb_cond_multiplier=ghb_fallback,
            recharge_year=year,
            rch_multiplier=1.0,
            **zone_kwargs
        )

        
        # Accept either 4-tuple or 5-tuple
        if isinstance(ret, tuple) and len(ret) == 5:
            head0, dem_tr, dem_crs, catch_poly, lst_path = ret
        else:
            head0, dem_tr, dem_crs, catch_poly = ret
            lst_path = None

        # Only print budget if lst_path is present
        if lst_path:
            print_global_budget_from_lst(lst_path)

    # Precise catchment mask from the polygon (create once after baseline succeeds)
    nrows, ncols = head0.shape
    catch_mask_poly = rasterize(
        [(catch_poly, 1)],
        out_shape=(nrows, ncols),
        transform=dem_tr,
        fill=0,
        all_touched=False,
        dtype="uint8"
    ).astype(bool)
   
   
    # 2) Patch in initial-head cache for subsequent runs
    install_initial_head_cache(
        catch_id=catch_id,
        year=year,
        dem_tr=dem_tr,
        dem_crs=dem_crs,
        model_shape=head0.shape,
        cache_root="data/output/cache"
    )
    print("Patch check:",
          sgd_utils.interpolate_well_heads,
          sys.modules['modflow_setup'].__dict__['interpolate_well_heads'])

    # 3) Observed heads ONCE (identical pipeline), using your helper with file cache
    obs_grid = load_or_interpolate_obs_heads(
    well_path   = filepaths['wells'],
    dem_path    = filepaths['dem'],
    catch_poly  = catch_poly,
    year        = year,
    dem_tr      = dem_tr,
    dem_crs     = dem_crs,
    model_shape = head0.shape,
    cache_dir   = os.path.join(filepaths['output'], "cache", str(catch_id), str(year)),  # <-- include year
    )

    

    print(f"[obs] year={year} valid cells={np.isfinite(obs_grid).sum()} "
      f"min/max={np.nanmin(obs_grid):.2f}/{np.nanmax(obs_grid):.2f}")

    # -- turn cached nodata values into NaN (common: -9999) --
    obs_grid = obs_grid.astype(float)
    obs_grid[~np.isfinite(obs_grid)] = np.nan
    obs_grid[obs_grid < -1000] = np.nan
    obs_grid[obs_grid > 5000]  = np.nan
    obs_grid[~catch_mask_poly] = np.nan     # <-- important: hide outside basin

    # quick diagnostics
    try:
        omin, omax = np.nanmin(obs_grid), np.nanmax(obs_grid)
        onum = np.isfinite(obs_grid).sum()
        print(f"[obs] valid cells: {onum}, min/max: {omin:.2f} … {omax:.2f} m")
    except ValueError:
        print("[obs] no valid cells after cleaning")
    
    wells_gdf = _wells_filtered_for_plot(filepaths['wells'], catch_poly, year, dem_crs)

    
    # Limit comparisons to cells reasonably supported by wells (e.g., 6 km radius)
    support_mask = make_well_support_mask(wells_gdf, dem_tr, head0.shape, radius_m=6000)
    comp_mask = catch_mask_poly & support_mask



    if wells_gdf is not None and not wells_gdf.empty:
        yrs = wells_gdf.get('date', pd.Series(pd.NaT, index=wells_gdf.index)).dt.year
        print(f"[wells] year={year} wells used={len(wells_gdf)} "
          f"unique years present={sorted(yrs.dropna().unique().tolist())[:10]}")
    else:
        print(f"[wells] year={year} no wells used")


    if obs_grid is None:
        print("⚠ No wells available for this catchment/year. Calibration aborted.")
        return
    obs_raw_tif = os.path.join(filepaths['output'], f"observed_heads_raw_c{catch_id}_y{year}.tif")
    profile = {
        "driver": "GTiff", "height": obs_grid.shape[0], "width": obs_grid.shape[1],
        "count": 1, "dtype": "float32", "crs": dem_crs, "transform": dem_tr,
        "compress": "lzw", "nodata": np.nan
    }
    with rasterio.open(obs_raw_tif, "w", **profile) as dst:
        dst.write(obs_grid.astype(np.float32), 1)
    print(f"✓ Saved raw observed heads (cleaned): {obs_raw_tif}")



    # 4) Prepare optimization
    
    # [soil, rock, riv, ghb, rch]
    #x0_full   = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    x0_full   = np.array([1.6570125272449432,
                      2.912898380647616,
                      1.0,
                      1.1,
                      1.1], dtype=float) #CHANGE TO YOUR PREFERRED START FROM TEH PREVIOUS RUN 
    #bounds_all = [(0.1, 3.0), (0.1, 3.0), (0.5, 2.0), (0.5, 2.0), (0.5, 2.0)]
    bounds_all = [
        (0.5, 2.5),  # soilK
        (0.5, 3.0),  # rockK
        (0.4, 1.5),  # river conductance (was tighter)
        (0.4, 1.5),  # ghb conductance
        (0.9, 1.1),  # recharge
    ]


    # collect fixed-from-CLI first
    
    active_idx = [0, 1, 2, 3, 4]
    fixed_vals = {}
    if args.fix_soil is not None: fixed_vals[0] = float(args.fix_soil)
    if args.fix_rock is not None: fixed_vals[1] = float(args.fix_rock)
    if args.fix_riv  is not None: fixed_vals[2] = float(args.fix_riv)
    if args.fix_ghb  is not None: fixed_vals[3] = float(args.fix_ghb)
    if args.fix_rch  is not None: fixed_vals[4] = float(args.fix_rch)

    # drop fixed indices from the active set (highest index first)
    for i in sorted(fixed_vals.keys(), reverse=True):
        if i in active_idx:
            active_idx.remove(i)

    x0 = x0_full[active_idx]
    bounds = [bounds_all[i] for i in active_idx]

    # helper to assemble full 4-vector from active x
    def full_params(x_active):
        x_full = x0_full.copy()
        # fill active
        for pos, i in enumerate(active_idx):
            x_full[i] = x_active[pos]
        # fill fixed
        for i, val in fixed_vals.items():
            x_full[i] = val
        return x_full

    
    # 4a) Coarse probe that respects fixed params (optional)
    if active_idx and (not args.skip_probe):
        objective_probe, _, _, _ = make_objective(
            catch_id=catch_id, year=year, mf6_exe=mf6_exe,
            filepaths=filepaths, obs_grid=obs_grid, catch_mask=comp_mask,
            stop_patience=9999, stop_tol=1e9  # effectively off
        )

        def probe_rmse(vec):
            return objective_probe(vec)

        start_full = full_params(x0)  # active from x0, fixed from fixed_vals
        cands = [start_full.copy()]
        for m in [0.8, 1.2]:
            cand = start_full.copy()
            for i in active_idx:     # only perturb the active ones
                cand[i] *= m
            cands.append(cand)

        probe_vals = []
        print("\n[probe] coarse sensitivity checks:")
        for v in cands:
            rmse_v = probe_rmse(v)
            print(f"  x={v}  → RMSE={rmse_v:.4f}")
            probe_vals.append((rmse_v, v))
        best_probe_rmse, best_probe_x = min(probe_vals, key=lambda t: t[0])
        print(f"[probe] best coarse start: {best_probe_x} (RMSE={best_probe_rmse:.4f})\n")

        # Use best coarse as starting point (restricted to active multipliers)
        x0 = best_probe_x[active_idx]

    # Real objective (with early stopping & best tracking)
    objective_fn, best, rmse_history, x_history = make_objective(
        catch_id=catch_id, year=year, mf6_exe=mf6_exe,
        filepaths=filepaths, obs_grid=obs_grid, catch_mask=comp_mask,
        stop_patience=8, stop_tol=1e-4, max_evals=args.maxiter
    )

    def objective_active(x_active):
        return objective_fn(full_params(x_active))
    
    # ---- single-run shortcut when nothing is free to optimize ----
    if len(active_idx) == 0:
        print("[opt] All parameters are fixed. Performing a single evaluation.")
        # Build objective just to compute one RMSE (and capture best['sim'])
        objective_fn, best, rmse_history, x_history = make_objective(
            catch_id=catch_id, year=year, mf6_exe=mf6_exe,
            filepaths=filepaths, obs_grid=obs_grid, catch_mask=comp_mask,
            stop_patience=1, stop_tol=0.0, max_evals=1
        )
        final_x = full_params(np.array([]))      # fills with fixed values
        final_rmse = objective_fn(final_x)       # one run

        # write a 1-row trace
        trace_csv = os.path.join(filepaths['output'], f"calib_trace_c{catch_id}_y{year}.csv")
        with open(trace_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter","rmse","soil_mult","rock_mult","riv_mult","ghb_mult","rch_mult"])
            w.writerow([1, final_rmse] + list(final_x))
        print(f"✓ Saved calibration trace: {trace_csv}")

        # write final params
        final_params_path = os.path.join(filepaths['output'], f"calib_final_params_c{catch_id}_y{year}.json")
        with open(final_params_path, "w") as jf:
            json.dump({
                "catchment": catch_id,
                "year": year,
                "soilK_multiplier": float(final_x[0]),
                "rockK_multiplier": float(final_x[1]),
                "riv_cond_multiplier": float(final_x[2]),
                "ghb_cond_multiplier": float(final_x[3]),
                "rch_multiplier": float(final_x[4]),
                "final_rmse": float(final_rmse),
                "iters": len(rmse_history)
            }, jf, indent=2)
        print(f"✓ Saved final parameters: {final_params_path}")

        # figures (re-use best['sim'] if captured; otherwise re-run once with final_x)
        if not args.no_figures:
            best_sim = best['sim']
            if best_sim is None:
                ret = setup_and_run_modflow(
                    catch_id, filepaths,
                    coastal_buffer=200.0,
                    mf6_exe=mf6_exe,
                    soilK_multiplier=final_x[0],
                    rockK_multiplier=final_x[1],
                    riv_cond_multiplier=final_x[2],
                    ghb_cond_multiplier=final_x[3],
                    recharge_year=year,
                    rch_multiplier=final_x[4],
                    **zone_kwargs
                )

                if isinstance(ret, tuple) and len(ret) == 5:
                    best_sim, _, _, _, lst_path = ret
                else:
                    best_sim, _, _, _ = ret
                    lst_path = None
                if lst_path:
                    print_global_budget_from_lst(lst_path)
            best_heads_tif = os.path.join(filepaths['output'], f"best_sim_heads_c{catch_id}_y{year}.tif")
            profile = {
                "driver": "GTiff",
                "height": best_sim.shape[0],
                "width": best_sim.shape[1],
                "count": 1,
                "dtype": "float32",
                "crs": dem_crs,
                "transform": dem_tr,
                "compress": "lzw",
                "nodata": np.nan
            }
            with rasterio.open(best_heads_tif, "w", **profile) as dst:
                dst.write(best_sim.astype(np.float32), 1)
            print(f"✓ Saved best simulated heads GeoTIFF: {best_heads_tif}")

            obs_heads_tif = os.path.join(filepaths['output'], f"observed_heads_clean_c{catch_id}_y{year}.tif")
            with rasterio.open(obs_heads_tif, "w", **profile) as dst:
                dst.write(obs_grid.astype(np.float32), 1)
            print(f"✓ Saved observed heads GeoTIFF: {obs_heads_tif}")

            plot_head_maps(obs_grid, best_sim, catch_id, year, filepaths['output'], catch_mask_poly)
            plot_rmse_convergence(rmse_history, catch_id, year, filepaths['output'])
            plot_wells_map(wells_gdf, catch_poly, dem_crs, catch_id, year, filepaths['output'])

        return  # end early; no SciPy optimizer needed


    # 5) Optimize (Powell + multistart; more robust on flat/noisy objectives)

    def _cb_and_print(x):
        # print accepted full params
        print_iter(full_params(x))
        # Early stop if last ~9 values are basically identical
        if len(rmse_history) >= 9:
            w = rmse_history[-9:]
            if (max(w) - min(w)) < 1e-4:
                print("↪ Early stop: RMSE flat window.")
                raise SystemExit("Early stop: RMSE flat")

    def _run_one_start(s):
        try:
            r = minimize(
                objective_active, s, bounds=bounds, method="Powell",
                callback=_cb_and_print,
                options={'maxiter': args.maxiter, 'ftol': args.ftol, 'xtol': 1e-3}
            )
            return r
        except SystemExit:
            return None

    # starters: current x0 plus 4 random points inside bounds
    starts = [x0.copy()]
    rng = np.random.default_rng(42)
    for _ in range(8):
        rnd = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)
        starts.append(rnd)

    best_res = None
    for s in starts:
        print("[multistart] trying start:", full_params(s))
        r = _run_one_start(s)
        if r is None:
            continue
        if (best_res is None) or (r.fun < best_res.fun):
            best_res = r

    if best_res is not None:
        res = best_res
        opt_x_active = res.x
        opt_fun = res.fun
    else:
        print("↪ No successful Powell runs; using starting point.")
        res = None
        opt_x_active = x0
        opt_fun = np.inf
        

    if best['x'] is not None and best['rmse'] < (opt_fun if np.isfinite(opt_fun) else np.inf):
        final_x = best['x']
        final_rmse = best['rmse']
    else:
        final_x = full_params(opt_x_active)
        final_rmse = float(opt_fun) if np.isfinite(opt_fun) else np.inf
    

    # ---- write calibration trace (RMSE + params per call) ----

    
    trace_csv = os.path.join(filepaths['output'], f"calib_trace_c{catch_id}_y{year}.csv")
    with open(trace_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter","rmse","soil_mult","rock_mult","riv_mult","ghb_mult","rch_mult"])
        for i, (rmse, x) in enumerate(zip(rmse_history, x_history), start=1):
            w.writerow([i, rmse] + list(x))
    print(f"✓ Saved calibration trace: {trace_csv}")

    
    # If SciPy didn't explore (x unchanged or too few evaluations), run a grid-search fallback
    
    sci_py_stagnant = (
    np.allclose(final_x, np.array([1.0, 1.0, 1.0, 1.0, 1.0]), rtol=0, atol=1e-6) or
    not np.isfinite(final_rmse) or
    final_rmse == np.inf
)
    
    
    if sci_py_stagnant and (not args.no_grid_fallback):
        print("↪ SciPy optimizer did not improve from the initial guess. Running fallback grid search…")
        grid_best = run_fallback_grid_search(catch_id, year, mf6_exe, filepaths, obs_grid, comp_mask)
        if grid_best['x'] is not None and grid_best['rmse'] < final_rmse:
            final_x = grid_best['x']
            final_rmse = grid_best['rmse']
            best = {'x': final_x, 'rmse': final_rmse, 'sim': grid_best['sim']}
    else:
        print("↪ Skipping grid-search fallback.")
    
    final_params_path = os.path.join(filepaths['output'], f"calib_final_params_c{catch_id}_y{year}.json")
    with open(final_params_path, "w") as jf:
        json.dump({
            "catchment": catch_id,
            "year": year,
            "soilK_multiplier": float(final_x[0]),
            "rockK_multiplier": float(final_x[1]),
            "riv_cond_multiplier": float(final_x[2]),
            "ghb_cond_multiplier": float(final_x[3]),
            "rch_multiplier": float(final_x[4]),
            "final_rmse": float(final_rmse),
            "iters": len(rmse_history)
        }, jf, indent=2)
    print(f"✓ Saved final parameters: {final_params_path}")






    # 6) Final run & outputs (ALWAYS re-run with final_x to ensure CBC/HDS match)
    ret = setup_and_run_modflow(
    
        catch_id, filepaths,
        coastal_buffer=200.0,
        mf6_exe=mf6_exe,
        soilK_multiplier=final_x[0],
        rockK_multiplier=final_x[1],
        riv_cond_multiplier=final_x[2],
        ghb_cond_multiplier=final_x[3],
        recharge_year=year,
        rch_multiplier=final_x[4],
        **zone_kwargs
    )
    if isinstance(ret, tuple) and len(ret) == 5:
        best_sim, _, _, _, lst_path = ret
    else:
        best_sim, _, _, _ = ret
        lst_path = None
    if lst_path:
        print_global_budget_from_lst(lst_path)

    # ---- save best simulated heads to GeoTIFF ----
    best_heads_tif = os.path.join(filepaths['output'], f"best_sim_heads_c{catch_id}_y{year}.tif")
    profile = {
        "driver": "GTiff",
        "height": best_sim.shape[0],
        "width": best_sim.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": dem_crs,      # from the baseline run above
        "transform": dem_tr, # from the baseline run above
        "compress": "lzw",
        "nodata": np.nan
    }
    with rasterio.open(best_heads_tif, "w", **profile) as dst:
        dst.write(best_sim.astype(np.float32), 1)
    print(f"✓ Saved best simulated heads GeoTIFF: {best_heads_tif}")

    # --- ALWAYS extract SGD (independent of --no-figures) ---
    try:
        from sgd_post import extract_sgd_from_cbc
        base_ws = os.path.join(filepaths['output'], f"model_runs/mf6_{catch_id}")
        sgd_csv = os.path.join(filepaths['output'], "sgd_summary.csv")
        res = extract_sgd_from_cbc(base_ws, catchment=catch_id, year=year, out_csv=sgd_csv)
        print(f"[SGD] {year}: {res['sgd_m3_per_day']:.3e} m³/d  ({res['sgd_m3_per_year']:.3e} m³/yr)")
    except FileNotFoundError as e:
        print(f"[SGD] CBC not found or unreadable: {e}")
    except Exception as e:
        print(f"[SGD] extraction failed: {e}")

    # --- Figures are optional ---
    if not args.no_figures:
        obs_heads_tif = os.path.join(filepaths['output'], f"observed_heads_clean_c{catch_id}_y{year}.tif")
        with rasterio.open(obs_heads_tif, "w", **profile) as dst:
            dst.write(obs_grid.astype(np.float32), 1)
        print(f"✓ Saved observed heads GeoTIFF: {obs_heads_tif}")

        plot_head_maps(obs_grid, best_sim, catch_id, year, filepaths['output'], catch_mask_poly)
        plot_rmse_convergence(rmse_history, catch_id, year, filepaths['output'])
        plot_wells_map(wells_gdf, catch_poly, dem_crs, catch_id, year, filepaths['output'])

    # 7) (Diagnostic) local 1D sweeps to verify flatness near final_x
    try:
        center = final_x.copy()
        var_names = ["soil_mult","rock_mult","riv_mult","ghb_mult","rch_mult"]
        grid = np.linspace(0.8, 1.5, 8)
        print("\n[diagnostic sweep] RMSE vs single-parameter sweeps around final_x:")
        for i, name in enumerate(var_names):
            vals = []
            for v in grid:
                x = center.copy(); x[i] = v
                rmse_v = objective_fn(x)
                vals.append(rmse_v)
            pairs = ", ".join(f"{g:.2f}:{rmse:.3f}" for g, rmse in zip(grid, vals))
            print(f"  {name}: {pairs}")
        print("")
    except Exception as _e:
        print(f"[diagnostic sweep] skipped: {_e}")

   
if __name__ == '__main__':
    main()