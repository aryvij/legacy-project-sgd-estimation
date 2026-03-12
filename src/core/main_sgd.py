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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import json

from core.flow_estimator import get_mean_discharge
from core.modflow_setup import setup_and_run_modflow

# Project directories
ROOT    = pathlib.Path(__file__).resolve().parents[2]
DATA    = ROOT / "data"
INPUT   = DATA / "input"
OUTPUT  = DATA / "output"

# Shapefiles and rasters
COAST_CHECK  = INPUT / "shapefiles/coastline_check/coastal_boundary.shp"
CATCH_SHP    = INPUT / "shapefiles/catchment/bsdbs.shp"
COAST_FOR_MF = INPUT / "shapefiles/coast_line/coastline.shp"

# Aquifer & conductivity datasets
SOIL_PERM    = INPUT / "aquifer_data" / "genomslapplighet" / "genomslapplighet.gpkg"
BEDROCK_K    = INPUT / "other_rasters" / "hydraulic_conductivity.tif"  # log10(K) raster

# Surface water shapefiles
RIVERS_SHP   = INPUT / "shapefiles" / "surface_water" / "Surface_water" /"hl_riks.shp" 
LAKES_SHP    = INPUT / "shapefiles" / "surface_water" / "scandinavian_waters_polygons" / "scandinavian_waters_polygons.shp"


def is_coastal(cid: str|int, buffer_m: float = 50.0) -> bool:
    cats = gpd.read_file(CATCH_SHP)[["ID_BSDB","geometry"]]
    poly = cats.loc[cats.ID_BSDB == int(cid), "geometry"].squeeze()
    if poly is None:
        raise ValueError(f"Catchment {cid} not found")
    coast = gpd.read_file(COAST_CHECK)
    if coast.crs != cats.crs:
        coast = coast.to_crs(cats.crs)
    coast_buf = coast.buffer(buffer_m)
    return coast_buf.unary_union.intersects(poly)


    parser = argparse.ArgumentParser(description="Driver for coastal SGD workflow")
    parser.add_argument("--catchment", type=int, help="Catchment ID (ID_BSDB)")
    parser.add_argument("--year", type=int, help="Year for simulation")
    parser.add_argument("--mf6", type=str, default=r"C:\Users\aryapv\AppData\Local\Programs\mf6.6.1_win64\bin\mf6.exe", help="Path to MODFLOW 6 executable")
    args = parser.parse_args()

    if args.catchment:
        cid = args.catchment
    else:
        try:
            cid = int(input("Enter catchment id (ID_BSDB): ").strip())
        except (ValueError, EOFError):
            print("⚠ Invalid catchment id or non-interactive mode detected.")
            sys.exit(1)

    if args.year:
        YEAR = args.year
    else:
        try:
            YEAR = int(input("Enter year for recharge/wells/sea-level (e.g., 2010): ").strip())
        except (ValueError, EOFError):
            print("⚠ Invalid year or non-interactive mode detected.")
            sys.exit(1)

    q, status = get_mean_discharge(cid)

    if not is_coastal(cid):
        print("→ Inland basin – no SGD simulation required.")
        return
    print("→ Coastal basin – running MODFLOW-6 for SGD …")

    # Year-specific recharge path
    per_year_rch = OUTPUT / "recharge_yearly" / f"recharge_egdi_gldas_{YEAR}.tif"
    if not per_year_rch.exists():
        print(f"⚠ Recharge raster not found:\n   {per_year_rch}")
        sys.exit(1)

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

    coastal_buffer = 200.0  # buffer for coastal GHB
    mf6_exe_path   = args.mf6

    heads, dem_tr, dem_crs, catch_poly = setup_and_run_modflow(
        catchment_id      = cid,
        filepaths         = filepaths,
        coastal_buffer    = coastal_buffer,
        mf6_exe           = mf6_exe_path,
        recharge_year     = YEAR,  # pass year into setup
        soilK_multiplier   =  1.4528859067902082,
        rockK_multiplier   = 2.9111852425870564,
        riv_cond_multiplier= 1.0624105361822733,
        ghb_cond_multiplier= 0.843504158508959,
        rch_multiplier     = 0.9068883707497266,
    )

    if heads is not None:
        fig, ax = plt.subplots(figsize=(6,5))
        vmin = np.nanmin(heads)
        vmax = np.nanmax(heads)
        print(f"Plotting heads from {vmin:.1f} to {vmax:.1f} m")
        im = ax.imshow(
            heads,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            origin="lower"
        )
        fig.colorbar(im, ax=ax, label="Simulated Head (m)")
        ax.set_title(f"Catchment {cid} – Final Head (Top Layer) – {YEAR}")
        plt.savefig(OUTPUT / f"heads_quicklook_{cid}_{YEAR}.png", dpi=150)
        plt.close(fig)
    else:
        print("⚠ MODFLOW run failed or returned no heads.")

if __name__ == "__main__":
    main()
