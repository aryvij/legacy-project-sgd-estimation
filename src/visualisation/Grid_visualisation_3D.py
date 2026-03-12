#!/usr/bin/env python3
# plot_idomain_3d_interactive.py
#
# Standalone script to:
#   • Read dem_clipped.tif & soil_thickness.tif for a given catchment
#   • Recreate id1/id2, form the “mode” array (0=inactive,1=rock only,2=rock+soil)
#   • Plot an interactive 3D surface of that mode array using Plotly
#   • Save a static PNG fallback in the same folder
#
# Usage:
#   python src/plot_idomain_3d_interactive.py
#   (Enter catchment ID when prompted)

import sys
import pathlib
import numpy as np
import rasterio
import plotly.graph_objs as go
import plotly.io as pio

def prompt_catchment_id():
    catch_id = input("Enter catchment ID: ").strip()
    if not catch_id.isdigit():
        print("Error: please enter a numeric catchment ID.")
        sys.exit(1)
    return catch_id

def load_dem_and_soil(base_ws: pathlib.Path):
    dem_path = base_ws / "dem_clipped.tif"
    sd_path  = base_ws / "soil_thickness.tif"
    if not dem_path.exists() or not sd_path.exists():
        print("Error: dem_clipped.tif or soil_thickness.tif not found in", base_ws)
        sys.exit(1)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        dem[dem == src.nodata] = np.nan

    with rasterio.open(sd_path) as src:
        sd = src.read(1).astype(float)
        sd[sd == src.nodata] = 0.0

    if sd.shape != dem.shape:
        print("Error: shapes mismatch between DEM and soil_thickness.")
        sys.exit(1)

    return dem, sd, dem.shape

def build_id_arrays(dem: np.ndarray, sd: np.ndarray):
    ROCK_BOTTOM_ELEV = -50.0
    valid = ~np.isnan(dem)
    botm1 = dem - sd

    id1 = np.zeros_like(dem, dtype=np.int8)
    id2 = np.zeros_like(dem, dtype=np.int8)

    id1[np.logical_and(valid, sd > 0.0)] = 1
    id2[np.logical_and(valid, botm1 > ROCK_BOTTOM_ELEV)] = 1

    return id1, id2

def build_mode_array(id1: np.ndarray, id2: np.ndarray):
    nrow, ncol = id1.shape
    mode = np.zeros((nrow, ncol), dtype=np.int8)

    both = np.logical_and(id1 == 1, id2 == 1)
    mode[both] = 2

    rock_only = np.logical_and(id1 == 0, id2 == 1)
    mode[rock_only] = 1

    # everything else remains 0
    return mode

def plotly_3d_surface(mode_array: np.ndarray, catch_id: str, out_folder: pathlib.Path):
    """
    Build an interactive Plotly 3D surface where Z = mode_array[i,j] (0,1,2).
    Also saves a static PNG snapshot to: idomain_3d_plotly_<catch_id>.png
    """
    nrow, ncol = mode_array.shape
    # X axis = columns (0 .. ncol-1), Y axis = rows (0 .. nrow-1)
    X = np.arange(ncol)
    Y = np.arange(nrow)
    Y, X = np.meshgrid(Y, X, indexing="ij")
    Z = mode_array  # each cell is exactly 0,1 or 2

    # Define a discrete colorscale: 
    #   0→light gray, 1→slate blue, 2→sandy brown
    colorscale = [
        [0.0, "#EEEEEE"],  # Z=0
        [0.333, "#EEEEEE"], 
        [0.333, "#6A5ACD"],  # Z=1
        [0.666, "#6A5ACD"],
        [0.666, "#C8A165"],  # Z=2
        [1.0, "#C8A165"],
    ]

    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=Z,          # color determined by category
        colorscale=colorscale,
        cmin=0,
        cmax=2,
        showscale=False,          # hide the discrete color bar
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Category: %{z}<extra></extra>"
    )

    layout = go.Layout(
        title=f"Interactive 3D IDomain (catchment {catch_id})",
        scene=dict(
            xaxis_title="Column (j)",
            yaxis_title="Row (i)",
            zaxis_title="Category",
            zaxis=dict(tickvals=[0,1,2], ticktext=["Inactive","Rock only","Rock+Soil"])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[surface], layout=layout)

    # 1) Save a static PNG fallback
    png_path = out_folder / f"idomain_3d_plotly_{catch_id}.png"
    # (this requires orca or kaleido; Plotly >= 4.9 includes kaleido by default)
    try:
        pio.write_image(fig, str(png_path), width=800, height=600, scale=1)
        print(f"  ✓ Saved static PNG: {png_path.name}")
    except Exception as e:
        print("  ⚠ Warning: could not save static PNG:", e)

    # 2) Launch browser for interactive view:
    print("Opening interactive 3D plot in browser…")
    fig.show()

def main():
    catch_id = prompt_catchment_id()

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
    base_ws = PROJECT_ROOT / "data" / "output" / "model_runs" / f"mf6_{catch_id}"
    print(f">>> plot_idomain_3d_interactive.py: Looking in:\n    {base_ws}\n")
    if not base_ws.exists():
        print(f"Error: folder not found:\n    {base_ws}")
        sys.exit(1)

    dem, sd, (nrow, ncol) = load_dem_and_soil(base_ws)
    id1, id2 = build_id_arrays(dem, sd)
    mode = build_mode_array(id1, id2)

    plotly_3d_surface(mode, catch_id, base_ws)
    print("\nDone. Check for the PNG and your browser window/tab for the interactive plot.")

if __name__ == "__main__":
    main()
