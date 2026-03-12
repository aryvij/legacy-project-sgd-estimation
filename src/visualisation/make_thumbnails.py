#!/usr/bin/env python3
# make_thumbnails.py
#
# Prompts for a numeric catchment ID, then looks in:
#   data/output/model_runs/mf6_<catchment_id>/
# for key “clipped” TIFFs (DEM, soil thickness, soil_k, rock_k, recharge, rivers, lakes).
# It writes out ~2×2 inch PNG thumbnails (≈200 px wide) alongside each TIFF.
#
# Usage:
#   cd <project_root>
#   python src/make_thumbnails.py
#   # (then type, e.g., 204, when prompted)

import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import rasterio

def prompt_catchment_id() -> str:
    """Ask for a numeric catchment ID; exit if invalid."""
    cid = input("Enter catchment ID: ").strip()
    if not cid.isdigit():
        print("⚠ Please enter a numeric catchment ID.")
        sys.exit(1)
    return cid

def tif_path(base_ws: pathlib.Path, name_no_ext: str) -> pathlib.Path:
    """Return Path to <name_no_ext>.tif under base_ws."""
    return base_ws / f"{name_no_ext}.tif"

def save_thumbnail(
    tpath: pathlib.Path,
    out_png: pathlib.Path,
    cmap: str = "viridis"
) -> None:
    """
    Load the single‐band raster at tpath, mask nodata→NaN,
    then write a small 2×2 inch PNG to out_png (≈200 px wide).
    """
    with rasterio.open(tpath) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

    valid = ~np.isnan(arr)
    if not np.any(valid):
        print(f"  → SKIPPING {tpath.name}: all values are nodata/NaN")
        return

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))

    fig, ax = plt.subplots(figsize=(2, 2))
    _ = ax.imshow(
        arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower"
    )
    ax.axis("off")
    plt.tight_layout(pad=0)

    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved thumbnail: {out_png.name}")

def main():
    catch_id = prompt_catchment_id()

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
    base_ws = PROJECT_ROOT / "data" / "output" / "model_runs" / f"mf6_{catch_id}"

    print(f"\n>>> make_thumbnails.py: looking in:\n    {base_ws}\n")
    if not base_ws.exists():
        print(f"Error: folder not found: {base_ws}")
        sys.exit(1)

    # Each tuple: (filename_stem without “.tif”, descriptive title, matplotlib colormap)
    expected = [
        ("dem_clipped",       "DEM (clipped)",          "terrain"),
        ("soil_thickness",    "Soil thickness",         "viridis"),
        ("soil_k",            "Soil permeability K",    "viridis"),
        ("rock_k",            "Bedrock conductivity K", "viridis"),
        ("recharge_clipped",  "Recharge (m/day)",       "YlGnBu"),
        ("rivers_cells",      "River mask",             "gray"),
        ("lakes_cells",       "Lake mask",              "gray"),
    ]

    for name_no_ext, title, cmap in expected:
        tpath = tif_path(base_ws, name_no_ext)
        if not tpath.exists():
            print(f"  → WARNING: {title} TIFF not found: {tpath.name}")
            continue

        out_png = base_ws / f"{name_no_ext}_thumb.png"
        save_thumbnail(tpath, out_png, cmap=cmap)

    print("\nAll done. You can now embed the *_thumb.png files on your slides.\n")

if __name__ == "__main__":
    main()
