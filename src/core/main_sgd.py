#!/usr/bin/env python3
# main_sgd.py
# Driver for coastal SGD workflow with two-layer MODFLOW
#Arya Vijayan 2025-09-30
# To run
# (env) PS C:\Users\aryapv\OneDrive - KTH\Modelling_SGD_Arya\SGD_model> python .\src\calibration_with_figures.py `      
# >>   --catchment 204 `
# >>   --year 2010 `
# >>   --mf6 "C:\Users\aryapv\AppData\Local\Programs\mf6.6.1_win64\bin\mf6.exe" `
# >>   --maxiter 1 `
# >>   --skip-probe `
# >>   --no-grid-fallback `
# >>   --fix-soil 1.4528859067902082 `
# >>   --fix-rock 2.9111852425870564 `  
# >>   --fix-riv  1.0624105361822733 `
# >>   --fix-ghb  0.843504158508959 `
# >>   --fix-rch  0.9068883707497266 `
# >>   --rch-elev-bins "0,10,30,60,200" `
# >>   --rch-elev-factors "1.02,1.1,1.16,1.20" `
# >>   --k-soil-factors   "1:1.40,2:0.80,3:0.95" `
# >>   --rch-soil-factors "1:0.80,2:1.10,3:1.05"

from __future__ import annotations
import os
import sys
import argparse
import pathlib

# Ensure the 'src' directory is on sys.path so "from core.…" imports work
# regardless of the working directory.
_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import json

from core.flow_estimator import get_mean_discharge
from core.modflow_setup import setup_and_run_modflow

# Default project directories (used when no CLI overrides are given)
_DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_INPUT  = _DEFAULT_ROOT / "data" / "input"
_DEFAULT_OUTPUT = _DEFAULT_ROOT / "data" / "output"


def is_coastal(cid: str|int, catch_shp=None, coast_check_shp=None, buffer_m: float = 50.0) -> bool:
    if catch_shp is None:
        catch_shp = _DEFAULT_INPUT / "shapefiles/catchment/bsdbs.shp"
    if coast_check_shp is None:
        coast_check_shp = _DEFAULT_INPUT / "shapefiles/coastline_check/coastal_boundary.shp"
    cats = gpd.read_file(catch_shp)[["ID_BSDB","geometry"]]
    poly = cats.loc[cats.ID_BSDB == int(cid), "geometry"].squeeze()
    if poly is None:
        raise ValueError(f"Catchment {cid} not found")
    coast = gpd.read_file(coast_check_shp)
    if coast.crs != cats.crs:
        coast = coast.to_crs(cats.crs)
    coast_buf = coast.buffer(buffer_m)
    return coast_buf.unary_union.intersects(poly)


def run_single_catchment(cid, YEAR, args, INPUT, OUTPUT):
    """Run a single catchment. Returns status string: 'success', 'inland', or 'failed: <reason>'."""
    CATCH_SHP    = INPUT / "shapefiles/catchment/bsdbs.shp"
    COAST_CHECK  = INPUT / "shapefiles/coastline_check/coastal_boundary.shp"
    COAST_FOR_MF = INPUT / "shapefiles/coast_line/coastline.shp"
    SOIL_PERM    = INPUT / "aquifer_data" / "genomslapplighet" / "genomslapplighet.gpkg"
    BEDROCK_K    = INPUT / "other_rasters" / "hydraulic_conductivity.tif"
    RIVERS_SHP   = INPUT / "shapefiles" / "surface_water" / "Surface_water" / "hl_riks.shp"
    LAKES_SHP    = INPUT / "shapefiles" / "surface_water" / "scandinavian_waters_polygons" / "scandinavian_waters_polygons.shp"

    try:
        if not is_coastal(cid, CATCH_SHP, COAST_CHECK):
            print(f"[{cid}] Inland basin – skipping.")
            return "inland"
    except Exception as e:
        print(f"[{cid}] ERROR in coastal check: {e}")
        return f"failed: coastal check error – {e}"

    print(f"[{cid}] Coastal basin – running MODFLOW-6 for SGD …")

    per_year_rch = OUTPUT / "recharge_yearly" / f"recharge_egdi_gldas_{YEAR}.tif"
    if not per_year_rch.exists():
        print(f"[{cid}] ERROR: Recharge raster not found: {per_year_rch}")
        return "failed: recharge raster missing"

    filepaths = {
        "dem"          : INPUT / "dem" / "elevation_sweden.tif",
        "catchment"    : CATCH_SHP,
        "recharge"     : per_year_rch,
        "soil_perm"    : SOIL_PERM,
        "soil_depth"   : INPUT / "aquifer_data" / "jorddjupsmodell" / "jorddjupsmodell_10x10m.tif",
        "conductivity" : BEDROCK_K,
        "sea_level"    : INPUT / "sea_level" / "yearly_average_sea_level.csv",
        "coast"        : COAST_FOR_MF,
        "wells"        : INPUT / "well_data" / "brunnar.gpkg",
        "rivers"       : RIVERS_SHP,
        "lakes"        : LAKES_SHP,
        "output"       : OUTPUT,
    }

    try:
        heads, dem_tr, dem_crs, catch_poly = setup_and_run_modflow(
            catchment_id       = cid,
            filepaths          = filepaths,
            coastal_buffer     = 200.0,
            mf6_exe            = args.mf6,
            recharge_year      = YEAR,
            soilK_multiplier   = 1.4528859067902082,
            rockK_multiplier   = 2.9111852425870564,
            riv_cond_multiplier= 1.0624105361822733,
            ghb_cond_multiplier= 0.843504158508959,
            rch_multiplier     = 0.9068883707497266,
            cell_size          = getattr(args, 'cell_size', None),
        )
    except Exception as e:
        print(f"[{cid}] ERROR during MODFLOW setup/run: {e}")
        return f"failed: {e}"

    if heads is None:
        print(f"[{cid}] No valid head results produced.")
        return "failed: no head output"

    fig, ax = plt.subplots(figsize=(6, 5))
    vmin, vmax = np.nanmin(heads), np.nanmax(heads)
    print(f"[{cid}] Plotting heads from {vmin:.1f} to {vmax:.1f} m")
    im = ax.imshow(heads, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    fig.colorbar(im, ax=ax, label="Simulated Head (m)")
    ax.set_title(f"Catchment {cid} – Final Head (Top Layer) – {YEAR}")
    plt.savefig(OUTPUT / f"heads_quicklook_{cid}_{YEAR}.png", dpi=150)
    plt.close(fig)
    return "success"


def main():
    parser = argparse.ArgumentParser(description="Driver for coastal SGD workflow")
    parser.add_argument("--catchment", type=int, help="Single catchment ID (ID_BSDB)")
    parser.add_argument("--catchments", type=str, default=None,
                        help="Comma-separated list of catchment IDs, or 'all' to run every coastal catchment")
    parser.add_argument("--year", type=int, help="Year for simulation")
    parser.add_argument("--mf6", type=str, default=r"C:\Users\aryapv\AppData\Local\Programs\mf6.6.1_win64\bin\mf6.exe", help="Path to MODFLOW 6 executable")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to the input data folder (default: <project>/data/input)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to the output folder (default: <project>/data/output)")
    parser.add_argument("--max-area", type=float, default=5000.0,
                        help="Skip catchments larger than this (km²). Default 5000.")
    parser.add_argument("--cell-size", type=float, default=None,
                        help="Resample DEM to this cell size (m). E.g. 200 for 200m grid. Default: use native DEM resolution.")
    args = parser.parse_args()

    # Resolve data paths from CLI or defaults
    INPUT  = pathlib.Path(args.data_root) if args.data_root else _DEFAULT_INPUT
    OUTPUT = pathlib.Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT

    # Resolve year
    if args.year:
        YEAR = args.year
    else:
        try:
            YEAR = int(input("Enter year for recharge/wells/sea-level (e.g., 2010): ").strip())
        except (ValueError, EOFError):
            print("⚠ Invalid year or non-interactive mode detected.")
            sys.exit(1)

    # Resolve catchment list
    CATCH_SHP = INPUT / "shapefiles/catchment/bsdbs.shp"
    COAST_CHECK = INPUT / "shapefiles/coastline_check/coastal_boundary.shp"

    if args.catchments:
        if args.catchments.strip().lower() == "all":
            cats = gpd.read_file(CATCH_SHP)
            # Only Swedish catchments (COUNTRY == "SW")
            sw = cats[cats["COUNTRY"] == "SW"].copy()
            all_ids = sorted(sw["ID_BSDB"].dropna().astype(int).unique().tolist())
            print(f"Found {len(all_ids)} Swedish catchments (filtered COUNTRY=='SW' from {len(cats)})")

            # Pre-filter to coastal only using a single spatial join (fast)
            print("Checking coastal intersection …")
            coast = gpd.read_file(COAST_CHECK)
            if coast.crs != sw.crs:
                coast = coast.to_crs(sw.crs)
            coast_buf = gpd.GeoDataFrame(geometry=coast.buffer(50), crs=sw.crs)
            coastal_join = gpd.sjoin(sw, coast_buf, how="inner", predicate="intersects")
            coastal_ids = set(coastal_join["ID_BSDB"].dropna().astype(int).unique())
            catchment_ids = sorted(cid for cid in all_ids if cid in coastal_ids)
            print(f"Found {len(catchment_ids)} Swedish coastal catchments")

            # Filter by max area
            if args.max_area:
                sw_indexed = sw.set_index("ID_BSDB")
                sw_indexed["area_km2"] = sw_indexed.geometry.area / 1e6
                before = len(catchment_ids)
                catchment_ids = [c for c in catchment_ids
                                 if c in sw_indexed.index and sw_indexed.loc[c, "area_km2"] <= args.max_area]
                skipped = before - len(catchment_ids)
                if skipped:
                    print(f"Skipped {skipped} catchments > {args.max_area} km² — {len(catchment_ids)} remaining")
        else:
            catchment_ids = [int(x.strip()) for x in args.catchments.split(",")]
    elif args.catchment:
        catchment_ids = [args.catchment]
    else:
        try:
            catchment_ids = [int(input("Enter catchment id (ID_BSDB): ").strip())]
        except (ValueError, EOFError):
            print("⚠ Invalid catchment id or non-interactive mode detected.")
            sys.exit(1)

    # Run each catchment (with resume support)
    import pandas as pd
    import time as _time

    summary_path = OUTPUT / f"batch_results_{YEAR}.xlsx"

    # Load previous results if resuming
    rows = []
    done_ids = set()
    if summary_path.exists():
        try:
            prev = pd.read_excel(summary_path)
            # Keep previously successful runs; re-run failures
            for _, r in prev.iterrows():
                if r["status"] == "success":
                    rows.append(r.to_dict())
                    done_ids.add(int(r["catchment_id"]))
            if done_ids:
                print(f"Resuming: {len(done_ids)} catchments already done — skipping them")
        except PermissionError:
            print("⚠ Cannot read batch_results Excel — is it open in another app? Close it and retry.")
            sys.exit(1)

    remaining = [c for c in catchment_ids if c not in done_ids]
    print(f"Catchments to run: {len(remaining)}  (total {len(catchment_ids)}, already done {len(done_ids)})")

    for i, cid in enumerate(remaining, 1):
        print(f"\n{'='*60}")
        print(f"  Catchment {cid}  ({i}/{len(remaining)},  overall {len(done_ids)+i}/{len(catchment_ids)})")
        print(f"{'='*60}")

        t0 = _time.time()
        status = run_single_catchment(cid, YEAR, args, INPUT, OUTPUT)
        elapsed = _time.time() - t0

        rows.append({
            "catchment_id": cid,
            "year": YEAR,
            "status": status,
            "runtime_sec": round(elapsed, 1),
        })

        # Save progress after each catchment (in case of crash)
        df = pd.DataFrame(rows)
        df.to_excel(summary_path, index=False)

    # Print summary
    df = pd.DataFrame(rows)
    n_success = (df["status"] == "success").sum()
    n_inland  = (df["status"] == "inland").sum()
    n_failed  = df["status"].str.startswith("failed").sum()

    print(f"\n{'='*60}")
    print(f"  BATCH SUMMARY  —  Year {YEAR}")
    print(f"{'='*60}")
    print(f"  Coastal catchments run: {len(catchment_ids)}")
    print(f"  Success:  {n_success}")
    print(f"  Inland:   {n_inland}")
    print(f"  Failed:   {n_failed}")
    if n_failed > 0:
        failed = df.loc[df["status"].str.startswith("failed"), "catchment_id"].tolist()
        print(f"  Failed IDs: {failed}")

    df.to_excel(summary_path, index=False)
    print(f"  Results saved to: {summary_path}")

if __name__ == "__main__":
    main()
