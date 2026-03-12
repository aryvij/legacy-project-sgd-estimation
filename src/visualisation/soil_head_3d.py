#!/usr/bin/env python3
# plot_head3d.py
#
# Usage: python plot_head3d.py
# (You will be prompted to enter the numeric catchment ID.)

import sys
import pathlib
import numpy as np
import rasterio
from rasterio.transform import Affine

# Try to import Plotly; if missing, prompt the user.
try:
    import plotly.graph_objects as go
except ImportError:
    print("Error: This script requires Plotly. Install it via `pip install plotly` or in your conda env.")
    sys.exit(1)


def prompt_catchment_id() -> str:
    """Prompt the user for a numeric catchment ID."""
    cid = input("Enter catchment ID: ").strip()
    if not cid.isdigit():
        print("Error: please enter a numeric catchment ID.")
        sys.exit(1)
    return cid


def get_head_tif_path(catch_id: str) -> pathlib.Path:
    """
    Construct the path to head_soil_<catch_id>.tif under data/output/model_runs/mf6_<catch_id>/.
    """
    project_root = pathlib.Path(__file__).resolve().parents[1]
    base_ws = project_root / "data" / "output" / "model_runs" / f"mf6_{catch_id}"
    tif_path = base_ws / f"head_soil_{catch_id}.tif"
    if not base_ws.exists():
        print(f"Error: model folder not found:\n  {base_ws}")
        sys.exit(1)
    if not tif_path.exists():
        print(f"Error: head_soil TIFF not found:\n  {tif_path}")
        sys.exit(1)
    return tif_path


def read_head_raster(tif_path: pathlib.Path):
    """
    Read the top-layer head array from the given GeoTIFF.
    Returns (z, transform, crs), where z is a 2D float array with NaNs for nodata.
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        transform = src.transform  # Affine
        crs = src.crs
    return arr, transform, crs


def make_xy_mesh(nrow: int, ncol: int, transform: Affine):
    """
    From an Affine transform and array shape, build two 2D arrays X, Y:
      X[i,j] = easting of cell center
      Y[i,j] = northing of cell center
    """
    # transform: Affine(a, b, c, d, e, f) where
    #   x_center = c + j*a + i*b  (but b,d are nearly 0 in north-up rasters)
    #   y_center = f + j*d + i*e
    # However, for a typical north-up DEM raster, b=d=0, a = pixel width, e = pixel height (negative).
    a = transform.a
    e = transform.e
    c = transform.c
    f = transform.f

    # Compute the x coordinates of each column’s center:
    #   j index runs 0..ncol-1; column center = c + (j + 0.5)*a
    x0 = c + 0.5 * a
    xs = x0 + np.arange(ncol) * a

    # Compute the y coordinates of each row’s center:
    #   i index runs 0..nrow-1; row center = f + (i + 0.5)*e
    y0 = f + 0.5 * e
    ys = y0 + np.arange(nrow) * e

    # But rasterio arrays are indexed arr[row, col] with row=0 at top, so the first element’s y = f + 0.5*e (with e negative).
    # Build meshgrid such that X,Y have shape (nrow,ncol):
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def plot_interactive_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, catch_id: str, out_folder: pathlib.Path):
    """
    Build an interactive Plotly 3D‐surface of Z(X,Y). Save to HTML.
    """
    # Mask NaNs so Plotly doesn’t attempt to render them:
    Z_masked = np.where(np.isnan(Z), None, Z)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z_masked,
                colorscale="Viridis",
                colorbar={"title": "Head (m)"},
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title=f"Interactive 3D Water-Table Head (Soil Layer) – Catchment {catch_id}",
        scene={
            "xaxis_title": "Easting",
            "yaxis_title": "Northing",
            "zaxis_title": "Head (m)",
            "aspectmode": "auto",
        },
        autosize=True,
    )

    out_html = out_folder / f"head3d_soil_{catch_id}.html"
    fig.write_html(str(out_html))
    print(f"✓ Saved interactive 3D surface to: {out_html.name}\n  (Open this file in your browser.)")


def main():
    catch_id = prompt_catchment_id()
    tif_path = get_head_tif_path(catch_id)

    # 1) Read the head raster
    Z, transform, crs = read_head_raster(tif_path)
    nrow, ncol = Z.shape

    # 2) Build X, Y meshes (in map coordinates)
    X, Y = make_xy_mesh(nrow, ncol, transform)

    # 3) Plot + save interactive 3D surface
    project_root = pathlib.Path(__file__).resolve().parents[1]
    base_ws = project_root / "data" / "output" / "model_runs" / f"mf6_{catch_id}"
    plot_interactive_3d(X, Y, Z, catch_id, base_ws)


if __name__ == "__main__":
    main()
