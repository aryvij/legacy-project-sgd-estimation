#!/usr/bin/env python3
# plot_idomain_bars.py
#
# Reads id1/id2 definitions for a given catchment run and:
#   • Builds a 2‐D “mode” array with categories:
#       0 = inactive, 1 = rock only, 2 = rock + soil
#   • Plots that 2‐D map with a 3‐color colormap
#   • Computes area of each category
#   • Draws a bar chart of “Area (km²)” per category
#
# Usage: python src/plot_idomain_bars.py
# (You will be prompted to enter the numeric catchment ID.)

import sys, os, pathlib
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ────────────────────────────────────────────────────────────────────────────────
def prompt_catchment_id():
    catch_id = input("Enter catchment ID: ").strip()
    if not catch_id.isdigit():
        print("Error: please enter a numeric catchment ID.")
        sys.exit(1)
    return catch_id

# ────────────────────────────────────────────────────────────────────────────────
def load_dem_and_soil(base_ws):
    """
    Open the clipped DEM and clipped soil thickness from GeoTIFFs:
      - dem_clipped.tif   → dem array (float), with np.nan where outside catchment
      - soil_thickness.tif→ sd array (float), with nodata replaced by np.nan or 0
    
    Return: dem (2D), sd_full (2D), transform (Affine), crs, nrow, ncol
    """
    dem_path = base_ws / "dem_clipped.tif"
    sd_path  = base_ws / "soil_thickness.tif"
    if not dem_path.exists() or not sd_path.exists():
        print("Error: dem_clipped.tif or soil_thickness.tif not found.")
        sys.exit(1)

    # 1) DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        dem[dem == src.nodata] = np.nan
        transform = src.transform
        crs = src.crs
    # 2) Soil thickness (sd_full)
    with rasterio.open(sd_path) as src:
        sd = src.read(1).astype(float)
        sd[sd == src.nodata] = 0.0
    if sd.shape != dem.shape:
        print("Error: soil_thickness shape does not match dem_clipped shape.")
        sys.exit(1)

    nrow, ncol = dem.shape
    return dem, sd, transform, crs, nrow, ncol

# ────────────────────────────────────────────────────────────────────────────────
def build_id_arrays(dem, sd):
    """
    Given dem (2D) and sd (2D) arrays, reconstruct:
       botm1 = dem - sd
       id1 = 1 where (dem is valid) AND (sd > 0), else 0
       id2 = 1 where (dem is valid) AND ((dem - sd) > ROCK_BOTTOM_ELEV), else 0

    NOTE: Must use the same ROCK_BOTTOM_ELEV as in your modflow_setup.
    """
    ROCK_BOTTOM_ELEV = -50.0

    valid = ~np.isnan(dem)
    botm1 = dem - sd

    id1 = np.zeros_like(dem, dtype=np.int8)
    id2 = np.zeros_like(dem, dtype=np.int8)

    # soil layer active if sd > 0 and dem not nan
    id1[np.logical_and(valid, sd > 0.0)] = 1
    # rock layer active if bottom of soil > ROCK_BOTTOM_ELEV
    id2[np.logical_and(valid, botm1 > ROCK_BOTTOM_ELEV)] = 1

    return id1, id2

# ────────────────────────────────────────────────────────────────────────────────
def build_mode_array(id1, id2):
    """
    Combine id1/id2 into a single 2D “mode” array:
      0 = inactive      (id1=0, id2=0)
      1 = rock only     (id1=0, id2=1)
      2 = rock + soil   (id1=1, id2=1)
    """
    nrow, ncol = id1.shape
    mode = np.zeros((nrow, ncol), dtype=np.int8)

    # Mark “rock + soil” first (id1=1 & id2=1)
    both = np.logical_and(id1 == 1, id2 == 1)
    mode[both] = 2

    # Mark “rock only” (id2=1 & id1=0)
    rock_only = np.logical_and(id1 == 0, id2 == 1)
    mode[rock_only] = 1

    # Everything else remains 0 (inactive)
    return mode

# ────────────────────────────────────────────────────────────────────────────────
def plot_mode_map(mode_array, transform, catch_id, out_png):
    """
    Plot the 2-D mode_array with a discrete 3-color colormap:
      0→inactive (light gray), 1→rock only (slate blue), 2→rock+soil (sandy brown).
    Save figure to out_png.
    """
    cmap = ListedColormap([
        "#EEEEEE",  # 0 = inactive
        "#2F4F4F",  # 1 = rock only (grey)
        "#C8A165",  # 2 = rock + soil (sandy brown)
    ])
    norm = BoundaryNorm([0, 1, 2, 3], cmap.N)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        mode_array,
        cmap=cmap,
        norm=norm,
        origin="lower"
    )
    ax.set_title(f"Layers (Catchment {catch_id})", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    # Build a legend by hand
    from matplotlib.patches import Patch
    patches = [
        Patch(facecolor="#EEEEEE", edgecolor="black", label="0: inactive"),
        Patch(facecolor="#6A5ACD", edgecolor="black", label="1: rock only"),
        Patch(facecolor="#C8A165", edgecolor="black", label="2: rock + soil"),
    ]
    ax.legend(
        handles=patches,
        loc="upper right",
        title="Legend",
        frameon=True,
        fontsize=10,
        title_fontsize=11
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved mode‐map as: {out_png}")

# ────────────────────────────────────────────────────────────────────────────────
def compute_category_areas(mode_array, transform):
    """
    Given mode_array, compute:
      - cell_area (m²) from transform
      - total area (m²) for each category 0,1,2

    Return a dict: {0: area0, 1: area1, 2: area2}
    """
    # Pixel dimensions from Affine transform:
    #    transform.a = pixel width (e.g. 10.0 m)
    #    transform.e = pixel height (negative if north‐up)
    cell_area = abs(transform.a * transform.e)

    areas = {}
    for cat in [0, 1, 2]:
        count = np.count_nonzero(mode_array == cat)
        areas[cat] = count * cell_area
    return areas

# ────────────────────────────────────────────────────────────────────────────────
def plot_bar_chart(areas_dict, catch_id, out_png):
    """
    Given a dict {0: area0, 1: area1, 2: area2}, draw a bar chart
    showing area in km² per category (0, 1, 2).
    """
    # Convert to km²
    cats = [0, 1, 2]
    values_m2 = [areas_dict[c] for c in cats]
    values_km2 = [v / 1e6 for v in values_m2]

    labels = [
        "Inactive",
        "Rock only",
        "Rock + Soil"
    ]
    colors = ["#EEEEEE", "#6A5ACD", "#C8A165"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values_km2, color=colors, edgecolor="black")
    ax.set_ylabel("Area (km²)")
    ax.set_title(f"Area by IDomain Category (Catchment {catch_id})")
    ax.set_ylim(0, max(values_km2) * 1.1)

    # Annotate each bar with its value (rounded)
    for idx, val in enumerate(values_km2):
        ax.text(idx, val + max(values_km2)*0.02, f"{val:.2f}", 
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved bar chart as: {out_png}")

# ────────────────────────────────────────────────────────────────────────────────
def main():
    catch_id = prompt_catchment_id()

    # Build path to the model‐run folder
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
    base_ws = PROJECT_ROOT / "data" / "output" / "model_runs" / f"mf6_{catch_id}"
    print(f">>> plot_idomain_bars.py: Looking in:\n    {base_ws}\n")
    if not base_ws.exists():
        print(f"Error: folder not found:\n    {base_ws}")
        sys.exit(1)

    # 1) Load DEM and soil‐thickness (both clipped to catchment)
    dem, sd, transform, crs, nrow, ncol = load_dem_and_soil(base_ws)

    # 2) Rebuild id1, id2 exactly as in modflow_setup logic
    id1, id2 = build_id_arrays(dem, sd)

    # 3) Build 3‐category “mode” array
    mode_array = build_mode_array(id1, id2)

    # 4) Plot the 2‐D mode map
    map_png = base_ws / f"idomain_map_{catch_id}.png"
    plot_mode_map(mode_array, transform, catch_id, out_png=map_png)

    # 5) Compute area (m²) of each category
    areas = compute_category_areas(mode_array, transform)

    # 6) Plot bar chart (area in km² vs category)
    bar_png = base_ws / f"idomain_barchart_{catch_id}.png"
    plot_bar_chart(areas, catch_id, out_png=bar_png)

    print("\nDone. Check the model_runs folder for the two PNGs.")

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
