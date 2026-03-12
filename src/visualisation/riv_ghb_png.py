#!/usr/bin/env python3
# plot_rivers_ghb.py
#
# Standalone script to plot:
#   • Catchment boundary
#   • Rivers within that catchment
#   • Coastal GHB cells (from the GHB mask raster)
#
# Saves a PNG “rivers_ghb_<catchment_id>.png” in the model folder.

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import box

def prompt_catchment_id():
    catch_id = input("Enter catchment ID: ").strip()
    if not catch_id.isdigit():
        print("Error: please enter a numeric catchment ID.")
        sys.exit(1)
    return catch_id

def get_paths(catch_id: str):
    """Return a dict of all required paths, verifying existence."""
    project_root = pathlib.Path(__file__).resolve().parents[1]
    paths = {}
    # 1) Catchment shapefile
    paths["catch_shp"] = project_root / "data" / "input" / "shapefiles" / "catchment" / "bsdbs.shp"
    # 2) Rivers shapefile (corrected to use scandinavian_waters_lines_shp folder)
    paths["rivers_shp"] = (
        project_root
        / "data"
        / "input"
        / "shapefiles"
        / "surface_water"
        / "scandinavian_waters_lines_shp"
        / "scandinavian_waters_lines.shp"
    )
    # 3) Base workspace for this catchment's model run
    base_ws = project_root / "data" / "output" / "model_runs" / f"mf6_{catch_id}"
    paths["base_ws"] = base_ws
    # 4) GHB mask raster
    paths["ghb_mask"] = base_ws / f"ghb_mask_{catch_id}.tif"

    # Verify existence
    if not paths["catch_shp"].exists():
        print(f"Error: catchment shapefile not found: {paths['catch_shp']}")
        sys.exit(1)
    if not paths["rivers_shp"].exists():
        print(f"Error: rivers shapefile not found: {paths['rivers_shp']}")
        sys.exit(1)
    if not base_ws.exists():
        print(f"Error: model folder not found: {base_ws}")
        sys.exit(1)
    if not paths["ghb_mask"].exists():
        print(f"Error: GHB mask raster not found: {paths['ghb_mask']}")
        sys.exit(1)
    return paths

def load_catchment_polygon(catch_shp: pathlib.Path, catch_id: str):
    """Read the catchment shapefile and return the single polygon for ID_BSDB == catch_id."""
    cats = gpd.read_file(catch_shp)[["ID_BSDB", "geometry"]]
    # Ensure ID_BSDB is string to match input
    cats["ID_BSDB"] = cats["ID_BSDB"].astype(str)
    single = cats[cats["ID_BSDB"] == catch_id]
    if single.empty:
        print(f"Error: catchment ID {catch_id} not found in {catch_shp.name}")
        sys.exit(1)
    poly = single.iloc[0].geometry
    crs = single.crs
    return poly, crs

def load_and_clip_rivers(rivers_shp: pathlib.Path, catch_poly, target_crs):
    """Read rivers shapefile, reproject if needed, clip to catchment polygon footprint."""
    rivers = gpd.read_file(rivers_shp)
    if rivers.crs != target_crs:
        rivers = rivers.to_crs(target_crs)
    # Clip to catchment polygon
    clipped = rivers[rivers.intersects(catch_poly)].copy()
    # Intersect ensures true geometry clipping
    clipped["geometry"] = clipped.geometry.intersection(catch_poly)
    return clipped

def read_ghb_mask(ghb_mask_path: pathlib.Path):
    """Open the GHB mask raster and return (array, transform, crs)."""
    with rasterio.open(ghb_mask_path) as src:
        arr = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs
        # treat nodata as zero/False
        if src.nodata is not None:
            arr[arr == src.nodata] = 0.0
    # create a boolean mask for any coastal‐GHB cell (value == 1)
    ghb_bool = (arr == 1.0)
    return ghb_bool, transform, crs

def plot_map(catch_poly, rivers_gdf, ghb_bool, ghb_tr, ghb_crs, catch_crs, catch_id, out_folder):
    """
    Build a single figure:
      - Catchment boundary (black outline)
      - Rivers (red lines)
      - GHB cells (filled blue squares)
    Save as “rivers_ghb_<catch_id>.png” in out_folder.
    """
    # Convert catchment polygon to GeoDataFrame to plot boundary
    catch_gdf = gpd.GeoDataFrame({"geometry": [catch_poly]}, crs=catch_crs)

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1) Plot GHB cells via imshow (extent from transform)
    nrow, ncol = ghb_bool.shape
    # Calculate extent: [xmin, xmax, ymin, ymax]
    xmin = ghb_tr.c
    xmax = xmin + ncol * ghb_tr.a
    ymax = ghb_tr.f
    ymin = ymax + nrow * ghb_tr.e

    # Create an array where GHB cells = 1, else 0
    plot_arr = np.zeros_like(ghb_bool, dtype=float)
    plot_arr[ghb_bool] = 1.0

    ax.imshow(
        plot_arr,
        cmap="Blues",
        vmin=0,
        vmax=1,
        extent=[xmin, xmax, ymin, ymax],
        origin="upper",
        alpha=0.6
    )

    # 2) Plot rivers (red lines)
    if not rivers_gdf.empty:
        rivers_gdf.plot(ax=ax, color="crimson", linewidth=1.0, label="Rivers")

    # 3) Plot catchment boundary (black outline)
    catch_gdf.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=1.5,
        label="Catchment Boundary"
    )

    # Legend styling
    from matplotlib.patches import Patch
    legend_entries = [
        Patch(facecolor="lightblue", edgecolor="none", label="Coastal cell"),
        Patch(facecolor="crimson", edgecolor="none", label="River segment"),
        Patch(facecolor="none", edgecolor="black", label="Catchment boundary"),
    ]
    ax.legend(handles=legend_entries, loc="lower left")

    ax.set_title(f"Catchment {catch_id}: Rivers & Coastal Cells", fontsize=16)
    #ax.set_xlabel("Easting")
    #ax.set_ylabel("Northing")
    ax.set_aspect("equal")

    # Save the figure
    out_path = out_folder / f"rivers_ghb_{catch_id}.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved figure: {out_path.name}")

def main():
    catch_id = prompt_catchment_id()
    paths = get_paths(catch_id)

    # 1) Load catchment polygon and its CRS
    catch_poly, catch_crs = load_catchment_polygon(paths["catch_shp"], catch_id)

    # 2) Load & clip rivers within that catchment
    rivers_clipped = load_and_clip_rivers(paths["rivers_shp"], catch_poly, catch_crs)

    # 3) Read GHB mask (boolean array) with its transform and CRS
    ghb_bool, ghb_tr, ghb_crs = read_ghb_mask(paths["ghb_mask"])

    # 4) If the GHB mask CRS differs, reproject it to catchment CRS
    if ghb_crs != catch_crs:
        print("Reprojecting GHB mask to catchment CRS…")
        from rasterio.warp import reproject, Resampling, calculate_default_transform

        # Compute new transform & dimensions
        new_tr, new_w, new_h = calculate_default_transform(
            ghb_crs,
            catch_crs,
            ghb_bool.shape[1],
            ghb_bool.shape[0],
            ghb_tr.c,
            ghb_tr.f + (ghb_bool.shape[0] * ghb_tr.e),
            ghb_tr.c + (ghb_bool.shape[1] * ghb_tr.a),
            ghb_tr.f
        )
        dst_arr = np.zeros((new_h, new_w), dtype=float)
        reproject(
            source=ghb_bool.astype(float),
            destination=dst_arr,
            src_transform=ghb_tr,
            src_crs=ghb_crs,
            dst_transform=new_tr,
            dst_crs=catch_crs,
            resampling=Resampling.nearest
        )
        ghb_bool = dst_arr.astype(bool)
        ghb_tr = new_tr
        ghb_crs = catch_crs

    # 5) Plot everything
    plot_map(
        catch_poly,
        rivers_clipped,
        ghb_bool,
        ghb_tr,
        ghb_crs,
        catch_crs,
        catch_id,
        paths["base_ws"]
    )

if __name__ == "__main__":
    main()