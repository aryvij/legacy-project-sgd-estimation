#!/usr/bin/env python3
# plot_results.py
#
# Usage: python src/plot_results.py
# (Enter the numeric catchment ID when prompted.)
#
# Expects all “clipped” TIFFs and the .cbc budget file under:
#   data/output/model_runs/mf6_<catchment_id>/
# Saves quick‐look PNGs back into that folder.

import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import flopy

# ────────────────────────────────────────────────────────────────────────────────
def prompt_catchment_id():
    """Ask the user for a numeric catchment ID."""
    catch_id = input("Enter catchment ID: ").strip()
    if not catch_id.isdigit():
        print("Error: please enter a numeric catchment ID.")
        sys.exit(1)
    return catch_id

# ────────────────────────────────────────────────────────────────────────────────
def tif_path(base_ws, name_no_ext):
    """Return Path to <name_no_ext>.tif under base_ws."""
    return base_ws / f"{name_no_ext}.tif"

# ────────────────────────────────────────────────────────────────────────────────
def find_lake_mask_tif(base_ws):
    """
    Prefer 'lake_chd_mask.tif'. If missing, look for 'lakes_cells.tif'.
    Return the name (without .tif) if found, else None.
    """
    for candidate in ("lake_chd_mask", "lakes_cells"):
        p = tif_path(base_ws, candidate)
        if p.exists():
            return candidate
    return None

# ────────────────────────────────────────────────────────────────────────────────
def plot_raster(base_ws, name_no_ext, title, cmap=None, unit_label=None):
    """
    If <name_no_ext>.tif exists under base_ws, load it, plot (imshow), and save as PNG.
    Returns True if plotted, False otherwise.
    """
    tpath = tif_path(base_ws, name_no_ext)
    if not tpath.exists():
        print(f"  → WARNING: {title} TIFF not found: {tpath.name}")
        return False

    with rasterio.open(tpath) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        valid = ~np.isnan(arr)
        if not np.any(valid):
            print(f"  → SKIPPING {title}: all values are NaN")
            return False

        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        fig, ax = plt.subplots(figsize=(6,5))
        chosen_cmap = cmap or ("viridis" if vmin >= 0 else "RdBu")
        im = ax.imshow(
            arr,
            cmap=chosen_cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower"
        )
        ax.set_title(f"{title}  (catchment {catch_id})", loc="right", fontsize=12, pad=6)
        cbar = fig.colorbar(im, ax=ax)
        label = title if unit_label is None else f"{title}  ({unit_label})"
        cbar.set_label(label)

        png_path = base_ws / f"{name_no_ext}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {title} as: {png_path.name}")
        return True

# ────────────────────────────────────────────────────────────────────────────────
def compute_flux_arrays(base_ws, catch_id, shape, transform):
    """
    Read gwf_<catch_id>.cbc and build two 2D arrays of shape=(nrow,ncol):
      • ghb_flux[i,j] = GHB flux at cell (i,j) [L³/day]
      • riv_flux[i,j] = RIV flux at cell (i,j) [L³/day]
    Also return total_ghb = sum of all GHB fluxes, and total_riv likewise.
    If “node”‐based records are present, convert node → (i,j).
    """
    cbc_path = base_ws / f"gwf_{catch_id}.cbc"
    if not cbc_path.exists():
        print(f"  → WARNING: Budget file not found: {cbc_path.name}")
        nrow, ncol = shape
        return np.zeros((nrow,ncol)), np.zeros((nrow,ncol)), 0.0, 0.0

    try:
        cbb = flopy.utils.CellBudgetFile(str(cbc_path))
    except Exception as e:
        print(f"  → WARNING: Could not open budget file: {e}")
        nrow, ncol = shape
        return np.zeros((nrow,ncol)), np.zeros((nrow,ncol)), 0.0, 0.0

    nrow, ncol = shape
    ghb_flux = np.zeros((nrow, ncol), dtype=float)
    riv_flux = np.zeros((nrow, ncol), dtype=float)
    total_ghb = 0.0
    total_riv = 0.0

    def extract_and_accumulate(tag, out_array):
        """
        Look for recarray(s) with text=tag (“GHB” or “RIV”). Take the last time step,
        locate either (i,j) or “node” field, fill out_array[i,j] = flow. Return sum(flow).
        """
        try:
            recs = cbb.get_data(text=tag)
        except Exception:
            print(f"  → WARNING: No records found for package '{tag}'")
            return 0.0

        if not recs:
            print(f"  → WARNING: No records found for package '{tag}'")
            return 0.0

        last = recs[-1]  # recarray for final time step
        names = last.dtype.names

        # Determine flow‐value field name:
        if "q" in names:
            flow_field = "q"
        elif "flow" in names:
            flow_field = "flow"
        elif "data" in names:
            flow_field = "data"
        else:
            print(f"  → WARNING: Cannot find flow field in '{tag}' records (fields: {names})")
            return 0.0

        # Case A: structured—has “i” and “j” columns
        if ("i" in names) and ("j" in names):
            total = 0.0
            for rec in last:
                i = int(rec["i"])
                j = int(rec["j"])
                f = float(rec[flow_field])
                out_array[i, j] = f
                total += f
            return total

        # Case B: structured but only “node” present
        if "node" in names:
            total = 0.0
            for rec in last:
                node = int(rec["node"])
                # Convert node → (layer, row, col).  Assume 0‐based:
                #   layer_index = node // (nrow*ncol)
                #   rem = node % (nrow*ncol)
                #   row = rem // ncol
                #   col = rem % ncol
                rem0 = node % (nrow * ncol)
                i = rem0 // ncol
                j = rem0 % ncol
                f = float(rec[flow_field])
                out_array[i, j] = f
                total += f
            return total

        print(f"  → WARNING: Cannot find (i,j) or node fields in '{tag}' records (fields: {names})")
        return 0.0

    total_ghb = extract_and_accumulate("GHB", ghb_flux)
    total_riv = extract_and_accumulate("RIV", riv_flux)

    return ghb_flux, riv_flux, total_ghb, total_riv

# ────────────────────────────────────────────────────────────────────────────────
def main():
    global catch_id
    catch_id = prompt_catchment_id()

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
    base_ws = PROJECT_ROOT / "data" / "output" / "model_runs" / f"mf6_{catch_id}"
    print(f">>> plot_results.py: Looking for data in:\n    {base_ws}\n")

    if not base_ws.exists():
        print(f"Error: Folder not found: {base_ws}")
        sys.exit(1)

    # 1) Clipped DEM
    plot_raster(
        base_ws, "dem_clipped",
        title="Clipped DEM",
        cmap="terrain",
        unit_label="m"
    )

    # 2) Soil Thickness
    plot_raster(
        base_ws, "soil_thickness",
        title="Soil Thickness",
        cmap="viridis",
        unit_label="m"
    )

    # 3) Soil Permeability (soil_k)
    plot_raster(
        base_ws, "soil_k",
        title="Soil Permeability K",
        cmap="viridis",
        unit_label="m/day"
    )

    # 4) Bedrock Conductivity (rock_k)
    plot_raster(
        base_ws, "rock_k",
        title="Bedrock Conductivity K",
        cmap="viridis",
        unit_label="m/day"
    )

    # 5) Head (Top/Soil Layer)
    plot_raster(
        base_ws, f"head_soil_{catch_id}",
        title="Hydraulic Head – Soil Layer",
        cmap="viridis",
        unit_label="m"
    )

    # 6) Head (Rock Layer)
    plot_raster(
        base_ws, f"head_rock_{catch_id}",
        title="Hydraulic Head – Rock Layer",
        cmap="viridis",
        unit_label="m"
    )

    # 7) Coastal GHB Mask
    plot_raster(
        base_ws, f"ghb_mask_{catch_id}",
        title="Coastal GHB Mask",
        cmap="Greys",
        unit_label="flag"
    )

    # 8) Inland CHD Mask
    plot_raster(
        base_ws, "chd_mask_inland_band",
        title="Inland CHD Mask",
        cmap="Greys",
        unit_label="flag"
    )

    # 9) Lake CHD Mask (try lake_chd_mask.tif, else lakes_cells.tif)
    lake_name = find_lake_mask_tif(base_ws)
    if lake_name:
        plot_raster(
            base_ws, lake_name,
            title="Lake CHD Mask",
            cmap="Greys",
            unit_label="flag"
        )
    else:
        print("  → WARNING: Neither lake_chd_mask.tif nor lakes_cells.tif found.")

    # 10) Build flux arrays from the .cbc file
    with rasterio.open(tif_path(base_ws, "dem_clipped")) as src:
        arr = src.read(1).astype(float)
        nrow, ncol = arr.shape
        transform = src.transform

    ghb_flux, riv_flux, total_ghb, total_riv = compute_flux_arrays(
        base_ws, catch_id, (nrow, ncol), transform
    )

    # 11) GHB Flux Distribution (fresh SGD)
    fig, ax = plt.subplots(figsize=(6,5))
     # Mask out zero‐flux cells so they appear white (or transparent)
    ghb_masked = np.ma.masked_where(ghb_flux == 0.0, ghb_flux)

    if np.any(~ghb_masked.mask):
        lim = np.max(np.abs(ghb_masked))
    else:
        lim = 1.0

    im = ax.imshow(
        ghb_masked,
        cmap="RdBu",     # blue = negative (outflow), red = positive (unlikely here)
        vmin=-lim,
        vmax= lim,
        origin="lower"
    )
    ax.set_title(f"GHB Flux (L³/day)  (catchment {catch_id})")
    fig.colorbar(im, ax=ax, label="GHB flux (L³/day)")

    png_path = base_ws / f"ghb_flux_{catch_id}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved GHB Flux (L³/day) as: {png_path.name}")

    # 12) RIV Flux Distribution (river–aquifer exchange)
    fig, ax = plt.subplots(figsize=(6,5))


    riv_masked = np.ma.masked_where(riv_flux == 0.0, riv_flux)
    if np.any(~riv_masked.mask):
        lim_riv = np.max(np.abs(riv_masked))
    else:
        lim_riv = 1.0

    im = ax.imshow(
        riv_masked,
        cmap="RdBu",
        vmin=-lim_riv,
        vmax= lim_riv,
        origin="lower"
    )
    ax.set_title(f"River–Aquifer Flux (L³/day)  (catchment {catch_id})")
    fig.colorbar(im, ax=ax, label="RIV flux (L³/day)")
    png_path = base_ws / f"riv_flux_{catch_id}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved River–Aquifer Flux (L³/day) as: {png_path.name}")

    

    # 13) Budget Bar Chart (|GHB| vs. |RIV|)
    fig, ax = plt.subplots(figsize=(4,4))
    labels = ["|GHB|", "|RIV|"]
    values = [abs(total_ghb), abs(total_riv)]
    bars = ax.bar(labels, values, color=["steelblue", "seagreen"])
    ax.set_ylabel("Total Flux (L³/day)")
    ax.set_title(f"Budget Summary  (catchment {catch_id})")
    for idx, val in enumerate(values):
        ax.text(idx, val * 1.02, f"{val:,.0f}", ha="center")
    png_path = base_ws / "budget_bar_chart.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved budget bar chart as: {png_path.name}")

    print("\nAll done. Check the folder for .png figures.")

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
