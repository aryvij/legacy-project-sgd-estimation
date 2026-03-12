#!/usr/bin/env python3
# plot_flux_patterns.py
#
# Standalone script to generate two PNGs:
#   • coastal_GHB_flux_<ID>.png
#   • river_flux_<ID>.png
#
# Each figure shows cells with flux values (│flux│ ≥ 10 % of the maximum absolute value)
# on the same grid as the clipped DEM. Negative values = blue (outflow to sea/river); 
# positive values = red (seawater intrusion or aquifer → river).
#
# Usage: python src/plot_flux_patterns.py
# (You will be prompted to enter a numeric catchment ID.)

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import flopy

def prompt_catchment_id():
    """Ask the user to type a numeric catchment ID and return it as a string."""
    catch_id = input("Enter catchment ID: ").strip()
    if not catch_id.isdigit():
        print("Error: please enter a numeric catchment ID.")
        sys.exit(1)
    return catch_id

def get_base_workspace(catch_id: str) -> pathlib.Path:
    """
    Construct and verify the path to data/output/model_runs/mf6_<catchment_id>
    """
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
    base_ws = PROJECT_ROOT / "data" / "output" / "model_runs" / f"mf6_{catch_id}"
    if not base_ws.exists():
        print(f"Error: model folder not found: {base_ws}")
        sys.exit(1)
    return base_ws

def read_flux_array(cbc_path: pathlib.Path, tag: str, shape: tuple[int,int]) -> np.ndarray:
    """
    From a .cbc (CellBudgetFile), extract the final time-step flux array for 'tag'
    (e.g. "GHB" or "RIV"). Returns a 2D NumPy array of shape (nrow,ncol)
    with flux values (positive or negative). If no records are found, returns zeros.
    """
    try:
        cbf = flopy.utils.CellBudgetFile(str(cbc_path))
    except Exception as e:
        print(f"→ Cannot open budget file: {e}")
        return np.zeros(shape, dtype=float)

    records = cbf.get_data(text=tag)
    if not records:
        print(f"→ No records found for '{tag}'")
        return np.zeros(shape, dtype=float)

    last = records[-1]  # final time step
    names = last.dtype.names

    # Determine which column holds the flux value
    if "q" in names:
        flux_field = "q"
    elif "flow" in names:
        flux_field = "flow"
    elif "data" in names:
        flux_field = "data"
    else:
        print(f"→ Cannot find flux column in '{tag}' records (fields: {names})")
        return np.zeros(shape, dtype=float)

    nrow, ncol = shape
    flux_arr = np.zeros((nrow, ncol), dtype=float)

    # Case 1: structured indexing (fields "i" and "j" exist)
    if ("i" in names) and ("j" in names):
        for rec in last:
            i = int(rec["i"])
            j = int(rec["j"])
            flux_arr[i, j] = float(rec[flux_field])
        return flux_arr

    # Case 2: “node” indexing → convert to (i, j)
    if "node" in names:
        for rec in last:
            node = int(rec["node"])
            rem0 = node % (nrow * ncol)
            i = rem0 // ncol
            j = rem0 % ncol
            flux_arr[i, j] = float(rec[flux_field])
        return flux_arr

    # If neither pattern matches:
    print(f"→ Cannot find (i,j) or node fields in '{tag}' records (fields: {names})")
    return flux_arr

def plot_coastal_ghb_flux(ghb_flux: np.ndarray, catch_id: str, out_folder: pathlib.Path):
    """
    Plot coastal GHB flux:
      - Negative (cold‐blue) = fresh groundwater → sea
      - Positive (red) = seawater intrusion
    Overlays filled circles on cells where │flux│ ≥ 10 % of the maximum absolute flux.
    Saves as "coastal_GHB_flux_<catchment_id>.png" in out_folder.
    """
    # Mask zeros so they don’t appear
    ghb_masked = np.ma.masked_where(ghb_flux == 0.0, ghb_flux)

    # Determine symmetric color bound (±lim)
    if np.any(~ghb_masked.mask):
        lim = np.max(np.abs(ghb_masked))
    else:
        lim = 1.0  # fallback if all zero

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        ghb_masked,
        origin="upper",   # <— use "upper" so row 0 is at top
        cmap="RdBu",
        vmin=-lim,
        vmax= lim,
    )
    ax.set_title(f"Coastal Flux (m³/day)  (catchment {catch_id})", fontsize=16)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, label="Flux\n(neg: GW → sea, pos: sea → GW)")
    ax.set_xticks([])
    ax.set_yticks([])

    # Overlay circles on “significant” flux cells (│flux│ ≥ 10% of lim)
    thresh = 0.10 * lim
    ys, xs = np.where(np.abs(ghb_masked.filled(0.0)) >= thresh)
    if ys.size > 0:
        vals = ghb_masked.data[ys, xs]
        abs_vals = np.abs(vals)
        # Circle size scaling: map [thresh … lim] to [min_size … max_size]
        max_size = 300
        min_size = 75
        sizes = (abs_vals / abs_vals.max()) * (max_size - min_size) + min_size

        from matplotlib.colors import Normalize
        norm = Normalize(vmin=-lim, vmax=lim)
        cmap = plt.get_cmap("RdBu")
        facecols = [cmap(norm(val)) for val in vals]

        ax.scatter(
            xs,
            ys,
            s=sizes,
            facecolors=facecols,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
        )

    plt.tight_layout()
    out_png = out_folder / f"coastal_GHB_flux_{catch_id}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✓ Saved: {out_png.name}")

def plot_river_flux_with_markers(riv_flux: np.ndarray, catch_id: str, out_folder: pathlib.Path):
    """
    Plot river–aquifer flux:
      - Negative (blue) = river → aquifer (“losing”)
      - Positive (red) = aquifer → river (“gaining”)
    Overlays filled circles on cells where │flux│ ≥ 10 % of the maximum absolute flux.
    Saves as "river_flux_<catchment_id>.png" in out_folder.
    """
    riv_masked = np.ma.masked_where(riv_flux == 0.0, riv_flux)

    if np.any(~riv_masked.mask):
        lim = np.max(np.abs(riv_masked))
    else:
        lim = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        riv_masked,
        origin="upper",   # <— use "upper" here as well
        cmap="RdBu",
        vmin=-lim,
        vmax= lim,
    )
    ax.set_title(f"River–Aquifer Flux (m³/day)  (catchment {catch_id})", fontsize=16)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, label="Flux\n(neg: R → GW, pos: GW → R)")
    ax.set_xticks([])
    ax.set_yticks([])

    # Overlay circles on “significant” flux cells (│flux│ ≥ 10% of lim)
    thresh = 0.10 * lim
    ys, xs = np.where(np.abs(riv_masked.filled(0.0)) >= thresh)
    if ys.size > 0:
        vals = riv_masked.data[ys, xs]
        abs_vals = np.abs(vals)
        max_size = 300
        min_size = 75
        sizes = (abs_vals / abs_vals.max()) * (max_size - min_size) + min_size

        from matplotlib.colors import Normalize
        norm = Normalize(vmin=-lim, vmax=lim)
        cmap = plt.get_cmap("RdBu")
        facecols = [cmap(norm(val)) for val in vals]

        ax.scatter(
            xs,
            ys,
            s=sizes,
            facecolors=facecols,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
        )

    plt.tight_layout()
    out_png = out_folder / f"river_flux_{catch_id}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✓ Saved: {out_png.name}")

def main():
    catch_id = prompt_catchment_id()
    base_ws = get_base_workspace(catch_id)

    # 1) Read the clipped DEM to get (nrow, ncol)
    dem_tif = base_ws / "dem_clipped.tif"
    if not dem_tif.exists():
        print(f"Error: clipped DEM not found at {dem_tif}")
        sys.exit(1)

    with rasterio.open(dem_tif) as src:
        nrow, ncol = src.read(1).shape

    # 2) Read GHB and RIV flux arrays from the .cbc budget file
    cbc_file = base_ws / f"gwf_{catch_id}.cbc"
    if not cbc_file.exists():
        print(f"Error: budget file not found: {cbc_file}")
        sys.exit(1)

    ghb_flux = read_flux_array(cbc_file, "GHB", (nrow, ncol))
    riv_flux = read_flux_array(cbc_file, "RIV", (nrow, ncol))

    # 3) Make the two PNG plots
    plot_coastal_ghb_flux(ghb_flux, catch_id, base_ws)
    plot_river_flux_with_markers(riv_flux, catch_id, base_ws)

if __name__ == "__main__":
    main()
