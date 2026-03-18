#!/usr/bin/env python3
import os, json, csv, numpy as np, pandas as pd, rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.ops import unary_union

from core.modflow_setup import setup_and_run_modflow, load_or_interpolate_obs_heads
from calibration.calibration_with_figures import plot_head_maps  # reuse your plotting

def run_validation(catch_id, calib_year, years_to_validate, mf6_exe, data_root=None, output_dir=None):
    input_dir  = data_root  or 'data/input'
    output_dir = output_dir or 'data/output'
    # load calibrated multipliers
    params_path = os.path.join(output_dir, f"calib_final_params_c{catch_id}_y{calib_year}.json")
    with open(params_path) as f:
        p = json.load(f)
    soilK, rockK, rivM, ghbM = p["soilK_multiplier"], p["rockK_multiplier"], p["riv_cond_multiplier"], p["ghb_cond_multiplier"]

    filepaths = {
        'dem':         os.path.join(input_dir, 'dem/elevation_sweden.tif'),
        'catchment':   os.path.join(input_dir, 'shapefiles/catchment/bsdbs.shp'),
        'recharge':    None,  # set per-year below
        'soil_perm':   os.path.join(input_dir, 'aquifer_data/genomslapplighet/genomslapplighet.gpkg'),
        'soil_depth':  os.path.join(input_dir, 'aquifer_data/jorddjupsmodell/jorddjupsmodell_10x10m.tif'),
        'conductivity':os.path.join(input_dir, 'other_rasters/hydraulic_conductivity.tif'),
        'sea_level':   os.path.join(input_dir, 'sea_level/yearly_average_sea_level.csv'),
        'coast':       os.path.join(input_dir, 'shapefiles/coast_line/coastline.shp'),
        'wells':       os.path.join(input_dir, 'well_data/brunnar.gpkg'),
        'rivers':      os.path.join(input_dir, 'shapefiles/surface_water/Surface_water/hl_riks.shp'),
        'lakes':       os.path.join(input_dir, 'shapefiles/surface_water/scandinavian_waters_polygons.shp'),
        'output':      output_dir
    }

    RECH_DIR = os.path.join(output_dir, 'recharge_yearly')
    RECH_BASENAME = "recharge_egdi_gldas_{year}.tif"

    rows = []
    for yy in years_to_validate:
        filepaths['recharge'] = os.path.join(RECH_DIR, RECH_BASENAME.format(year=yy))
        if not os.path.exists(filepaths['recharge']):
            print(f"[skip] missing recharge for {yy}: {filepaths['recharge']}")
            continue

        # 1) baseline run once to get grid and polygon
        head0, dem_tr, dem_crs, catch_poly = setup_and_run_modflow(
            catch_id, filepaths,
            coastal_buffer=200.0, mf6_exe=mf6_exe,
            soilK_multiplier=soilK, rockK_multiplier=rockK,
            riv_cond_multiplier=rivM, ghb_cond_multiplier=ghbM,
            recharge_year=yy
        )

        nrows, ncols = head0.shape
        catch_mask_poly = rasterize([(catch_poly,1)],
            out_shape=(nrows,ncols), transform=dem_tr,
            fill=0, all_touched=False, dtype="uint8").astype(bool)

        # 2) observed heads for that year (cached)
        obs_grid = load_or_interpolate_obs_heads(
            well_path   = filepaths['wells'],
            dem_path    = filepaths['dem'],
            catch_poly  = catch_poly,
            year        = yy,
            dem_tr      = dem_tr,
            dem_crs     = dem_crs,
            model_shape = head0.shape,
            cache_dir   = os.path.join(filepaths['output'],"cache",str(catch_id),str(yy))
        ).astype(float)

        # same cleaning as your calibrator
        for arr in (head0, obs_grid):
            arr[~np.isfinite(arr)] = np.nan
        obs_grid[(obs_grid < -1000) | (obs_grid > 5000)] = np.nan
        head0[(head0 < -1000) | (head0 > 5000)] = np.nan
        obs_grid[~catch_mask_poly] = np.nan
        head0[~catch_mask_poly] = np.nan

        mask = catch_mask_poly & np.isfinite(head0) & np.isfinite(obs_grid)
        rmse = np.sqrt(np.nanmean((head0[mask] - obs_grid[mask])**2)) if np.any(mask) else np.inf

        rows.append([yy, soilK, rockK, rivM, ghbM, int(mask.sum()), float(rmse)])

        # residual maps / figures
        plot_head_maps(obs_grid, head0, catch_id, yy, filepaths['output'], catch_mask_poly)

    # write validation summary
    out_csv = os.path.join(output_dir, f"validation_summary_c{catch_id}_cal{calib_year}.csv")
    with open(out_csv,"w",newline="") as f:
        w = csv.writer(f); w.writerow(["year","soilK","rockK","rivM","ghbM","n_cells","rmse"])
        w.writerows(rows)
    print(f"✓ Saved validation summary: {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--catchment", "-c", type=int, required=True)
    ap.add_argument("--calib-year", "-y", type=int, required=True, help="year you calibrated")
    ap.add_argument("--years", "-Y", type=int, nargs="+", required=True, help="years to validate")
    ap.add_argument("--mf6", required=True)
    ap.add_argument("--data-root", type=str, default=None,
                    help="Path to input data folder (default: data/input)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Path to output folder (default: data/output)")
    args = ap.parse_args()
    run_validation(args.catchment, args.calib_year, args.years, args.mf6,
                   data_root=args.data_root, output_dir=args.output_dir)
