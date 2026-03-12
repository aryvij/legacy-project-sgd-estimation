# File: src/sgd_utils.py
# 2025-09-30 Arya Vijayan


import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata


def load_and_mask_raster(raster_path, geometry, target_crs):
    """Load a raster, reproject if needed, then mask to 'geometry'."""
    with rasterio.open(raster_path) as src:
        if src.crs != target_crs:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            data = np.empty((height, width), dtype=src.dtypes[0])
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )

            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(**kwargs) as dataset:
                    dataset.write(data, 1)
                    out_image, out_transform = mask(dataset, geometry, crop=True)
        else:
            out_image, out_transform = mask(src, geometry, crop=True)

        return out_image[0], out_transform, src.nodata


def resample_to_target(src_array, src_transform, src_crs,
                       target_shape, target_transform, target_crs,
                       resampling=Resampling.bilinear):
    """Resample src_array to match the shape/transform of the target."""
    dst_array = np.empty(target_shape, dtype=src_array.dtype)
    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=resampling
    )
    return dst_array


def interpolate_well_heads(
    well_gpkg,
    catchment_geom,
    target_shape,
    target_transform,
    target_crs,
    dem_array,
    year: int | None = None
) -> np.ndarray:
    """
    Interpolate well heads for a given calibration year (or all data if year=None).

    - Parses date from 'nivadatum' or 'borrdatum'.
    - Filters to the specified `year` if provided.
    - Clips wells to the catchment polygon.
    - Uses 'grundvattenniva' as measured head.
    - Interpolates head = DEM - depth.
    - Fills gaps with nearest-neighbor.
    - Final fallback: head = 0.98 * DEM everywhere if no valid wells.
    """
    wells = gpd.read_file(well_gpkg)

    # 1) parse date
    if 'nivadatum' in wells.columns:
        wells['date'] = pd.to_datetime(
            wells['nivadatum'].astype(str),
            format='%Y%m%d', errors='coerce'
        )
    elif 'borrdatum' in wells.columns:
        wells['date'] = pd.to_datetime(
            wells['borrdatum'].astype(str),
            format='%Y%m%d', errors='coerce'
        )
    else:
        wells['date'] = pd.NaT

    # 2) filter by year
    if year is not None:
        wells = wells[wells['date'].dt.year == year]

    # 3) clip to catchment & reproject
    if wells.crs != target_crs:
        wells = wells.to_crs(target_crs)
    wells = wells[wells.geometry.intersects(catchment_geom[0])].copy()

    # 4) grab numeric head
    wells['level'] = pd.to_numeric(wells.get('grundvattenniva', None),
                                   errors='coerce')
    wells = wells.dropna(subset=['level'])

    # if no valid wells at all, fallback everywhere
    if wells.empty:
        return 0.98 * dem_array

    # 5) only keep point geometries
    wells = wells[wells.geometry.geom_type == 'Point']
    if wells.empty:
        return 0.98 * dem_array

    wells['x'] = wells.geometry.x
    wells['y'] = wells.geometry.y

    pts = wells[['x','y']].to_numpy()
    vals = wells['level'].to_numpy(dtype=float)

    # 6) grid for interpolation
    nrow, ncol = target_shape
    xs = np.linspace(
        target_transform[2] + target_transform[0]/2,
        target_transform[2] + target_transform[0]*(ncol - 0.5),
        ncol
    )
    ys = np.linspace(
        target_transform[5] + target_transform[4]/2,
        target_transform[5] + target_transform[4]*(nrow - 0.5),
        nrow
    )
    grid_x, grid_y = np.meshgrid(xs, ys)

    # 7) linear interp + nearest‐neighbor fill
    head_depth = griddata(pts, vals, (grid_x, grid_y), method='linear')
    mask_nan = np.isnan(head_depth)
    if mask_nan.any():
        head_depth[mask_nan] = griddata(pts, vals,
                                        (grid_x, grid_y),
                                        method='nearest')[mask_nan]

    # 8) compute head = DEM – depth, cap at DEM
    head = dem_array - head_depth
    head = np.minimum(head, dem_array)

    # 9) final fallback where still NaN
    still_nan = np.isnan(head)
    if still_nan.any():
        head[still_nan] = 0.98 * dem_array[still_nan]

    return head

def create_coastal_mask(coast_shp, catchment_geom, buffer_width,
                        target_shape, target_transform, target_crs):
    """Create coastal buffer zone. True= inside buffer."""
    coast = gpd.read_file(coast_shp)
    if coast.crs != target_crs:
        coast = coast.to_crs(target_crs)

    coast_buffer = coast.buffer(buffer_width)
    coastal_union = coast_buffer.unary_union.intersection(catchment_geom[0])
    coastal_zone = gpd.GeoDataFrame({'geometry': [coastal_union]}, crs=target_crs)

    nrow, ncol = target_shape
    xs = np.linspace(
        target_transform[2] + target_transform[0]/2,
        target_transform[2] + target_transform[0]*(ncol - 0.5),
        ncol
    )
    ys = np.linspace(
        target_transform[5] + target_transform[4]/2,
        target_transform[5] + target_transform[4]*(nrow - 0.5),
        nrow
    )
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = [Point(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel())]
    pts_gdf = gpd.GeoDataFrame({'geometry': pts}, crs=target_crs)
    join = gpd.sjoin(pts_gdf, coastal_zone, how='left', predicate='within')
    coastal_mask = ~join.index_right.isna().values
    return coastal_mask.reshape(target_shape)


def create_surface_mask(shp_path, catchment_geom,
                        target_shape, target_transform, target_crs):
    """Create boolean mask array for lakes/rivers shapefile intersection."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    geom_union = gdf.unary_union.intersection(catchment_geom[0])
    gdf_masked = gpd.GeoDataFrame({'geometry': [geom_union]}, crs=target_crs)

    nrow, ncol = target_shape
    xs = np.linspace(
        target_transform[2] + target_transform[0]/2,
        target_transform[2] + target_transform[0]*(ncol - 0.5),
        ncol
    )
    ys = np.linspace(
        target_transform[5] + target_transform[4]/2,
        target_transform[5] + target_transform[4]*(nrow - 0.5),
        nrow
    )
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = [Point(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel())]
    pts_gdf = gpd.GeoDataFrame({'geometry': pts}, crs=target_crs)
    join = gpd.sjoin(pts_gdf, gdf_masked, how='left', predicate='intersects')
    mask_bool = ~join.index_right.isna().values
    return mask_bool.reshape(target_shape)


def save_array_as_geotiff(array, out_path, transform, crs,
                          nodata_val=-9999.0, unit_name=None):
    """Save a 2D NumPy array as a GeoTIFF with optional unit metadata."""
    arr_to_write = array.astype(np.float32, copy=True)
    arr_to_write[np.isnan(arr_to_write)] = nodata_val

    height, width = arr_to_write.shape
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': nodata_val,
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform
    }

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(arr_to_write, 1)
        if unit_name:
            dst.update_tags(units=unit_name)

    msg_unit = f" (units={unit_name})" if unit_name else ""
    print(f"Saved raster to {out_path}{msg_unit}")
