#!/usr/bin/env python3
# grid_info.py
#
# Standalone script to report:
#   • MODFLOW grid cell size (delr, delc) [m]
#   • Total number of cells in the catchment (non‐NaN pixels)
#
# Usage: python grid_info.py
# (You will be prompted to enter the numeric catchment ID.)

import sys
import pathlib
import numpy as np
import rasterio


def prompt_catchment_id() -> str:
    """Ask the user for a numeric catchment ID."""
    cid = input("Enter catchment ID: ").strip()
    if not cid.isdigit():
        print("Error: please enter a numeric catchment ID.")
        sys.exit(1)
    return cid


def main():
    catch_id = prompt_catchment_id()

    # Locate the clipped DEM for this catchment
    project_root = pathlib.Path(__file__).resolve().parents[1]
    dem_path = project_root / "data" / "output" / "model_runs" / f"mf6_{catch_id}" / "dem_clipped.tif"

    if not dem_path.exists():
        print(f"Error: clipped DEM not found at:\n  {dem_path}")
        sys.exit(1)

    # Open the raster and read metadata
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        transform = src.transform
        nodata = src.nodata

    # Replace nodata values with numpy.nan for masking
    if nodata is not None:
        dem[dem == nodata] = np.nan

    # Compute grid spacing from the Affine transform:
    #   delr = transform.a    (pixel width, in map units)
    #   delc = abs(transform.e) (pixel height, in map units)
    delr = transform.a
    delc = abs(transform.e)

    # Count how many “active” cells there are (non‐NaN)
    valid_mask = ~np.isnan(dem)
    n_active = int(np.count_nonzero(valid_mask))
    n_total = dem.size

    print("\n── Grid Information ─────────────────────────────────────────────────\n")
    print(f"Catchment ID           : {catch_id}")
    print(f"Clipped DEM path       : {dem_path.name}")
    print(f"Grid cell size (delr) : {delr:.2f} m")
    print(f"Grid cell size (delc) : {delc:.2f} m")
    print(f"Total cells in raster  : {n_total}")
    print(f"Active (non‐NaN) cells : {n_active}")
    print(f"Inactive (NaN) cells   : {n_total - n_active}")
    print("\n──────────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
