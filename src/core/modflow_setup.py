
#!/usr/bin/env python3
# modflow_setup.py
# 2025-09-30 Arya Vijayan
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import rasterize
import flopy
from scipy.interpolate import griddata
from shapely.geometry import Point
from shapely.ops import unary_union
import datetime
from scipy.ndimage import binary_erosion
from flopy.mf6.modflow.mfgwfdrn import ModflowGwfdrn
from scipy.ndimage import distance_transform_edt




from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling


print("[modflow_setup] using file:", __file__)
import inspect
print("[modflow_setup] rasterize ref:", inspect.getsource(rasterize)[:60], "...")


# Utility functions imported from sgd_utils
from core.sgd_utils import (
    load_and_mask_raster,
    resample_to_target,
    interpolate_well_heads,
    create_coastal_mask,
    save_array_as_geotiff,
)

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
MIN_SOIL_THICKNESS = 0.2    # bare rock thickness
ROCK_BOTTOM_ELEV   = -50.0   # bottom of fractured rock (m asl)
SOIL_NODATA        = -9999   # nodata for soil thickness
# Permeability classes 1–3 mapped to m/s (Larsson 2008 mid-values)
CLASS_K_M_PER_S   = {1:1e-8, 2:1e-6, 3:1e-5}

# Where to keep per-catchment cached rasters
CACHE_ROOT = os.path.join("data", "output", "cache")
# ────────────────────────────────────────────────────────────────────────────────

def load_clean_tif(path):
    """Load 1‐band GeoTIFF as float64, turn its nodata → np.nan."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.transform, src.crs

def clip_minimum(arr, minimum=1e-9):
    """Enforce numeric floor on a conductivity array."""
    return np.where(np.isnan(arr) | (arr <= 0), minimum, arr)

def print_time(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Observed heads file-level cache (safe: only used if called by calibrator)
# ─────────────────────────────────────────────────────────────────────────────
def load_or_interpolate_obs_heads(
    well_path: str,
    dem_path: str,
    catch_poly,
    year: int,
    dem_tr,
    dem_crs,
    model_shape,
    cache_dir: str,
):
    """
    Build observed head grid using the SAME logic as initial heads (depth→head via
    sgd_utils.interpolate_well_heads with DEM on the model grid). Cache to GeoTIFF
    so later runs just read from disk.

    Returns: 2D NumPy array (float32/float64), shape=model_shape.
    """
    
    

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"observed_heads_{year}.tif")

    # Reuse cached file if present
    if os.path.exists(cache_file):
        with rasterio.open(cache_file) as src:
            arr = src.read(1)
        print(f"[cache] Reusing observed heads from {cache_file}")
        return arr

    # Reproject DEM onto the exact model grid (transform/CRS/shape)
    nrows, ncols = model_shape
    with rasterio.open(dem_path) as src:
        with WarpedVRT(
            src,
            crs=dem_crs,
            transform=dem_tr,
            width=ncols,
            height=nrows,
            resampling=Resampling.bilinear,
        ) as vrt:
            dem_grid = vrt.read(1).astype(float)
            if vrt.nodata is not None:
                dem_grid[dem_grid == vrt.nodata] = np.nan

    # Interpolate wells using the SAME function as for initial heads
    obs_grid = interpolate_well_heads(
        well_gpkg        = well_path,
        catchment_geom   = [catch_poly],
        target_shape     = model_shape,
        target_transform = dem_tr,
        target_crs       = dem_crs,
        dem_array        = dem_grid,
        year             = year,
    )

    # Save to GeoTIFF for persistent reuse
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'crs': dem_crs,
        'transform': dem_tr,
        'height': model_shape[0],
        'width':  model_shape[1],
        'nodata': -9999.0,
    }
    out = obs_grid.astype('float32').copy()
    out[~np.isfinite(out)] = -9999.0
    with rasterio.open(cache_file, 'w', **profile) as dst:
        dst.write(out, 1)
    print(f"[cache] Saved observed heads to {cache_file}")
    return obs_grid


def _resample_cached(cache_path, nrow, ncol, dem_tr, dem_crs, method=Resampling.nearest):
    """Load a cached raster and resample to model grid if shapes differ."""
    with rasterio.open(cache_path) as src:
        if src.height == nrow and src.width == ncol:
            arr = src.read(1).astype(float)
            nd = src.nodata
            if nd is not None:
                arr[arr == nd] = np.nan
            return arr
        with WarpedVRT(src, crs=dem_crs, transform=dem_tr,
                       width=ncol, height=nrow,
                       resampling=method) as vrt:
            arr = vrt.read(1).astype(float)
            nd = vrt.nodata
            if nd is not None:
                arr[arr == nd] = np.nan
        return arr


def setup_and_run_modflow(
    catchment_id: int,
    filepaths: dict[str, str],
    coastal_buffer: float,
    mf6_exe: str,
    soilK_multiplier: float = 1.0,
    rockK_multiplier: float = 1.5,
    riv_cond_multiplier: float = 0.6,
    ghb_cond_multiplier: float = 1.5,
    recharge_year: int | None = None,
    rch_multiplier: float = 0.95,
    rch_elev_bins: list[float] | None = None,
    rch_elev_factors: list[float] | None = None,
    rch_soil_factors: dict[int, float] | None = None,
    k_soil_factors: dict[int, float] | None = None,
    cell_size: float | None = None,
) -> tuple[np.ndarray, rasterio.Affine, object, object]:
    print_time("Start setup_and_run_modflow")

    # 1) Load catchment polygon
    print_time("Loading catchment polygon")
    cats = gpd.read_file(filepaths['catchment'])[['ID_BSDB', 'geometry']]
    cats['ID_BSDB'] = pd.to_numeric(cats['ID_BSDB'], errors='coerce')
    sel = cats[cats.ID_BSDB == catchment_id]
    if sel.empty:
        raise RuntimeError(f"Catchment {catchment_id} not found")
    dem_crs = rasterio.open(filepaths['dem']).crs
    sel = sel.to_crs(dem_crs)
    catch_poly = unary_union(sel.geometry)

    # ── Set up cache folder ────────────────────────────────────────────────
    cache_root = os.path.join(str(filepaths.get("output", "data/output")), "cache")
    cache_dir = os.path.join(cache_root, str(catchment_id))
    os.makedirs(cache_dir, exist_ok=True)

    # 2) DEM: clip & cache
    print_time("Processing DEM")
    dem_cache = os.path.join(cache_dir, 'dem_clipped.tif')
    if not os.path.exists(dem_cache):
        with rasterio.open(filepaths['dem']) as src:
            dem_img, dem_tr = mask(src, [catch_poly], crop=True)
            profile = src.profile.copy()
            profile.update({
                'driver': 'GTiff',
                'height': dem_img.shape[1],
                'width': dem_img.shape[2],
                'transform': dem_tr
            })
        with rasterio.open(dem_cache, 'w', **profile) as dst:
            dst.write(dem_img)
    dem, dem_tr, dem_crs = load_clean_tif(dem_cache)

    # ── Optional grid coarsening ───────────────────────────────────────────
    if cell_size and cell_size > dem_tr.a:
        from rasterio.warp import Resampling as _Res
        from rasterio.vrt import WarpedVRT
        print_time(f"Resampling DEM from {dem_tr.a:.0f}m to {cell_size:.0f}m")
        with rasterio.open(dem_cache) as src_dem:
            new_width  = max(1, int(src_dem.width  * dem_tr.a / cell_size))
            new_height = max(1, int(src_dem.height * abs(dem_tr.e) / cell_size))
            new_tr = rasterio.transform.from_bounds(
                *src_dem.bounds, new_width, new_height)
            with WarpedVRT(src_dem, width=new_width, height=new_height,
                           transform=new_tr, resampling=_Res.average) as vrt:
                dem = vrt.read(1).astype(float)
                nd = vrt.nodata
                if nd is not None:
                    dem[dem == nd] = np.nan
        dem_tr = new_tr
        _coarsened = True
        print_time(f"Resampled grid: {dem.shape[0]} rows x {dem.shape[1]} cols")
    else:
        _coarsened = False

    catch_mask = ~np.isnan(dem)

    # Guard: empty grid (catchment outside DEM extent or fully nodata)
    if catch_mask.sum() == 0:
        print(f"[SKIP] Catchment {catchment_id}: DEM clip produced zero valid cells")
        return None, None, None, None
    
    # and *immediately* dump for viz:
    dem_viz = dem.copy()
    dem_viz[~catch_mask] = np.nan
    save_array_as_geotiff(
        dem_viz,
        os.path.join(cache_dir, 'dem_for_viz.tif'),
        dem_tr, dem_crs
    )
    nrow, ncol = dem.shape
    cell_area = dem_tr.a * abs(dem_tr.e)

    print_time(f"Grid dimensions: rows={nrow}, cols={ncol}")

    # 3) Soil-depth: clip/warp, process, cache
    print_time("Processing soil depth")
    sd_cache = os.path.join(cache_dir, 'soil_depth.tif')
    if not os.path.exists(sd_cache):
        with rasterio.open(filepaths['soil_depth']) as src_sd:
            with WarpedVRT(src_sd, crs=dem_crs, transform=dem_tr,
                           width=ncol, height=nrow,
                           resampling=Resampling.nearest) as vrt:
                sd = vrt.read(1).astype(float)
                sd[sd == vrt.nodata] = np.nan
            profile = src_sd.profile.copy()
            profile.update({
                'driver': 'GTiff',
                'height': nrow,
                'width': ncol,
                'transform': dem_tr
            })
        sd[np.isnan(dem)] = np.nan
        coords = np.column_stack(np.where(~np.isnan(sd)))
        vals = sd[~np.isnan(sd)]
        mask_fill = np.isnan(sd) & ~np.isnan(dem)
        sd[mask_fill] = griddata(coords, vals,
                                  np.column_stack(np.where(mask_fill)),
                                  method='nearest')
        sd[np.isnan(sd)] = MIN_SOIL_THICKNESS
        with rasterio.open(sd_cache, 'w', **profile) as dst:
            dst.write(sd[np.newaxis, ...])
    sd = _resample_cached(sd_cache, nrow, ncol, dem_tr, dem_crs, method=Resampling.nearest)
    sd_viz = sd.copy()
    sd_viz[~catch_mask] = np.nan
    save_array_as_geotiff(
        sd_viz, 
        os.path.join(cache_dir, 'soil_depth_for_viz.tif'),
        dem_tr,
        dem_crs
)
    
        # --- ensure L1 bottom never goes below rock bottom ---
    bot1_raw = dem - sd                      # raw L1 bottom
    BOT2 = np.full_like(dem, ROCK_BOTTOM_ELEV)
    # clamp L1 bottom to be at least ROCK_BOTTOM_ELEV + 0.1 m
    bot1= np.maximum(bot1_raw, ROCK_BOTTOM_ELEV + 0.1)
    # define effective soil thickness consistent with clamped bottom
    eff_sd = np.maximum(dem - bot1, 0.0)

    # 4) Recharge: csv or raster -> cache
    year_tag = str(recharge_year) if recharge_year is not None else "raster"
    rech_cache = os.path.join(cache_dir, f"recharge_{year_tag}.tif")

    # build one DEM‐based profile
    with rasterio.open(dem_cache) as ref:
        profile = ref.profile.copy()
    profile.update({
        "driver":"GTiff","dtype":"float32","count":1,
        "height":nrow,"width":ncol,
        "transform":dem_tr,"nodata":-9999.0
    })
    force_rebuild = bool(int(os.environ.get("FORCE_RECHARGE_REBUILD", "0")))
    src_mtime = os.path.getmtime(filepaths['recharge']) if os.path.exists(filepaths['recharge']) else 0.0
    cache_mtime = os.path.getmtime(rech_cache) if os.path.exists(rech_cache) else -1.0
    if force_rebuild or (cache_mtime < src_mtime):
        if os.path.exists(rech_cache):
            print(f"[recharge] Source is newer or FORCE set → deleting cache: {rech_cache}")
            os.remove(rech_cache)
    if not os.path.exists(rech_cache):
        if str(filepaths['recharge']).lower().endswith('.csv') and recharge_year is not None:
            # CSV interpolation
            df = pd.read_csv(filepaths['recharge'], sep=';', engine='python')  # adjust sep if needed
            df_y = df[df["year"] == recharge_year].dropna(subset=["lon", "lat", "Recharge_mm_year"])
            if df_y.empty:
                raise ValueError(f"No recharge points found for year {recharge_year}")
            gdf = gpd.GeoDataFrame(
                df_y.copy(),
                geometry=gpd.points_from_xy(df_y.lon, df_y.lat),
                crs="EPSG:4326"   # adjust if your CSV is not WGS84
            ).to_crs(dem_crs)

            # Use projected x/y (meters) for interpolation
            pts = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
            vals_r = gdf["Recharge_mm_year"].values / 1000.0 / 365.0  # mm/yr → m/day

            # Model grid in same CRS (already correct)
            xs = np.linspace(dem_tr.c, dem_tr.c + dem_tr.a*(ncol-1), ncol)
            ys = np.linspace(dem_tr.f, dem_tr.f + dem_tr.e*(nrow-1), nrow)
            gx, gy = np.meshgrid(xs, ys)

            # Interpolate
            rech = griddata(pts, vals_r, (gx, gy), method='linear')
            mask_nan = np.isnan(rech)
            rech[mask_nan] = griddata(pts, vals_r, (gx, gy), method='nearest')[mask_nan]
        else:
            # Raster recharge (unit-aware)
            with rasterio.open(filepaths['recharge']) as src_r:
                path_abs = os.path.abspath(filepaths['recharge'])
                print(f"[recharge] opened raster: {path_abs}")
                print(f"[recharge] raster size={src_r.width}x{src_r.height} res={src_r.res} crs={src_r.crs} dtype={src_r.dtypes[0]}")
                tags = {k.lower(): v for k, v in src_r.tags().items()}

            img, tr, nd = load_and_mask_raster(filepaths['recharge'], [catch_poly], dem_crs)
            # turn nodata values into NaN before any stats/conversions
            if nd is not None and np.isfinite(nd):
                img = np.where(img == nd, np.nan, img)

            # resample to model grid if needed
            if img.shape != dem.shape:
                img = resample_to_target(img, tr, dem_crs, dem.shape, dem_tr, dem_crs)

            # Inspect raw values before any conversion
            raw_min = float(np.nanmin(img))
            raw_max = float(np.nanmax(img))
            print(f"[recharge] raw (as read) min/max = {raw_min:.3e}/{raw_max:.3e}")

            # Decide units
            units = (tags.get('units','') or tags.get('unit','')).lower()
            if 'm/day' in units or 'm per day' in units or 'm/d' in units:
                print("[recharge] units tag indicates m/day -  no conversion")
                rech = img.astype(float)
            else:
                # Heuristic: if values are tiny (<< 1), likely already m/day
                if raw_max < 0.02:  # <2 cm/day is a typical recharge upper bound
                    print("[recharge] guessing m/day from value range → no conversion")
                    rech = img.astype(float)
                else:
                    print("[recharge] assuming mm/yr → converting to m/day (÷1000÷365)")
                    rech = img / 1000.0 / 365.0

            # Final sanity after conversion
            mn = float(np.nanmin(rech)); mx = float(np.nanmax(rech))
            mu = float(np.nanmean(rech)); sdv = float(np.nanstd(rech))
            print(f"[recharge] IN-BASIN (m/day): min={mn:.3e} max={mx:.3e} mean={mu:.3e} std={sdv:.3e}")
            print(f"[recharge] mean ≈ {mu*365*1000:.1f} mm/yr")

            rech = np.clip(rech, 0, None)
            rech[np.isnan(dem)] = np.nan

            
        with rasterio.open(rech_cache, 'w', **profile) as dst:
            dst.write(rech.astype(np.float32)[np.newaxis, ...])

    # load & inspect (resample if grid was coarsened)
    rech = _resample_cached(rech_cache, nrow, ncol, dem_tr, dem_crs, method=Resampling.average)
    print(f">>> recharge min/max: {np.nanmin(rech):.2e}, {np.nanmax(rech):.2e}")
    rech[~catch_mask] = np.nan
    save_array_as_geotiff(
        rech,
        os.path.join(cache_dir, f"recharge_{year_tag}_for_viz.tif"),
        dem_tr, dem_crs
    )
    
    # 5) Conductivity: soil_k & rock_k

   

    print_time("Processing conductivity")

    soil_cache = os.path.join(cache_dir, 'soil_class.tif')

    if not os.path.exists(soil_cache):
        print("[soil] building soil_class.tif")
        perm = gpd.read_file(filepaths['soil_perm'], layer='genomslapplighet').to_crs(dem_crs)

        # pick the right attribute
        colname = ('genomslapp' if 'genomslapp' in perm.columns
                else 'genomslapplighet' if 'genomslapplighet' in perm.columns
                else None)
        if colname is None:
            raise RuntimeError("Soil-perm layer is missing 'genomslapp'/'genomslapplighet'")

        perm[colname] = pd.to_numeric(perm[colname], errors='coerce')
        perm = perm[perm.geometry.notnull() & (~perm.geometry.is_empty)].copy()
        # fix invalid geometries if any
        bad = ~perm.geometry.is_valid
        if bad.any():
            perm.loc[bad, 'geometry'] = perm.loc[bad, 'geometry'].buffer(0)

        # integer class 0..3
        perm['__class__'] = (perm[colname].fillna(0).round(0).astype('int64').clip(0, 3))
        print(f"[soil] unique classes in vector: {sorted(perm['__class__'].unique().tolist())}")

        # only write >0 (1..3); 0 is fill
        shapes = [(g, int(v)) for g, v in zip(perm.geometry, perm['__class__']) if int(v) > 0]

        soil_class_arr = rasterize(
            shapes=shapes,
            out_shape=(nrow, ncol),
            transform=dem_tr,
            fill=0,
            dtype='int16',
            all_touched=False
        )

        # save cache
        with rasterio.open(os.path.join(cache_dir, 'dem_clipped.tif')) as ref:
            profile = ref.profile.copy()
        profile.update({'driver':'GTiff', 'dtype':'int16', 'count':1, 'nodata':0})
        with rasterio.open(soil_cache, 'w', **profile) as dst:
            dst.write(soil_class_arr, 1)

    # ---- ALWAYS reload here (outside the if) ----
    if not os.path.exists(soil_cache):
        raise RuntimeError(f"[soil] expected {soil_cache} but it was not created.")

    soil_class = _resample_cached(soil_cache, nrow, ncol, dem_tr, dem_crs, method=Resampling.nearest).astype(np.int16)

    print(f"[soil] soil_class raster dtype={soil_class.dtype}, "
        f"min={int(soil_class.min())}, max={int(soil_class.max())}")

    # map classes -> K (m/s -> m/day)
    soil_k = np.full((nrow, ncol), np.nan, dtype=np.float64)
    for cls, k_ms in CLASS_K_M_PER_S.items():
        soil_k[soil_class == cls] = k_ms * 86400.0  # m/s -> m/day

    # Fallback for unmapped (class 0) inside catchment: use class-1 instead of near-impermeable
    soil_k[(soil_class == 0) & catch_mask] = CLASS_K_M_PER_S[1] * 86400.0  # ~0.864 m/day

    # Keep outside catchment as NaN; apply multiplier
    
    if k_soil_factors:
        for cls, fac in k_soil_factors.items():
            m = (soil_class == int(cls)) & catch_mask
        soil_k[m] *= float(fac)
    soil_k[~catch_mask] = np.nan
    soil_k *= soilK_multiplier
    print(">>> soil_k min/max:", np.nanmin(soil_k), np.nanmax(soil_k))
    
    
    save_array_as_geotiff(soil_k, os.path.join(cache_dir, 'soil_k_for_viz.tif'), dem_tr, dem_crs)
    print_time("Saved soil_k_for_viz.tif")


    print_time(f"Loading conductivity raster ...")
    img_c, tr_c, nd_c = load_and_mask_raster(filepaths['conductivity'], [catch_poly], dem_crs)
    print_time(f"Resampling conductivity raster to match DEM ...")
    if img_c.shape != dem.shape:
        img_c = resample_to_target(img_c, tr_c, dem_crs, dem.shape, dem_tr, dem_crs)
    print_time(f"Calculating rock hydraulic conductivity ...")
    rock_k = np.where(img_c==nd_c, np.nan, (10.0**img_c)*86400.0)
    print_time(f"Applying nodata and multiplier to rock_k ...")
    rock_k = np.where(np.isnan(rock_k)|(rock_k<=0), 1e-9, rock_k) * rockK_multiplier

    rock_k = clip_minimum(rock_k, minimum=1e-9)
    rock_k[~catch_mask] = np.nan
    save_array_as_geotiff(
        rock_k, 
        os.path.join(cache_dir, 'rock_k_for_viz.tif'), 
        dem_tr, 
        dem_crs
)
    
    # 6) Initial heads — start close to DEM to help Newton convergence
    print_time("Interpolating initial heads")
    try:
        h0 = interpolate_well_heads(filepaths['wells'], [catch_poly], dem.shape, dem_tr, dem_crs, dem, year=recharge_year)
    except Exception as e:
        print(f"[init heads] Well interpolation failed ({e}) — using DEM-based fallback")
        h0 = None
    if h0 is None:
        h0 = dem - 5.0   # conservative: water table 5 m below ground
    h0[np.isnan(h0)] = dem[np.isnan(h0)] - 5.0   # ungauged cells: 5 m below DEM
    h0 = np.minimum(h0, dem + 0.5)   # max 0.5 m above ground (artesian limit)
    h0 = np.maximum(h0, dem - 50.0)  # don't go below rock bottom
    # In coastal/low-lying cells, clamp head to at least sea level
    h0 = np.where(dem < 5.0, np.maximum(h0, 0.0), h0)
    h0[~catch_mask] = np.nan
    print(f"[init heads] min={np.nanmin(h0):.1f} max={np.nanmax(h0):.1f} m")
    save_array_as_geotiff(
        h0, 
        os.path.join(cache_dir, 'initial_head_for_viz.tif'), 
        dem_tr, 
        dem_crs
    )

    # --- build & run MF6 ---
    print_time("Building and running MODFLOW 6 model")
    base_ws = os.path.join(filepaths['output'], f"model_runs/mf6_{catchment_id}")
    os.makedirs(base_ws, exist_ok=True)
    sim = flopy.mf6.MFSimulation(sim_name=f"sgd_{catchment_id}", exe_name=mf6_exe, sim_ws=base_ws,
                                  continue_=True)  # continue past non-converging time steps
    periods = [
    (30.0,  30, 1.0),   # (perlen, nstp, tsmult)
    (335.0, 50, 1.0),
    ]
    flopy.mf6.ModflowTdis(sim, nper=2, time_units='days', perioddata=periods)
    flopy.mf6.ModflowIms(sim, print_option='ALL', complexity='MODERATE',
                         linear_acceleration='BICGSTAB',
                         inner_maximum=1000, inner_dvclose=1e-5,
                         outer_maximum=500, outer_dvclose=1e-4,
                         relaxation_factor=0.0,
                         under_relaxation='DBD',
                         under_relaxation_theta=0.5,
                         under_relaxation_kappa=0.05,
                         under_relaxation_gamma=0.0,
                         backtracking_number=20,
                         csv_output_filerecord='ims.csv')
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=f"gwf_{catchment_id}",
        save_flows=True,
        newtonoptions="UNDER_RELAXATION",  # Newton with damping
    )
    from scipy.ndimage import label

    
    
    active1_base = ((~np.isnan(dem)) & (eff_sd > 0))                  # layer 1 active
    # Keep L2 active anywhere inside the catchment (now safe because bot1 >= ROCK_BOTTOM_ELEV+0.1)
    active2_base = (~np.isnan(dem))                         

    # remove tiny L1 islands
    lbl, ncomp = label(active1_base)
    MIN_SIZE = 1  # don't prune L1 islands; prevents holes in idomain
    small = np.zeros_like(active1_base, dtype=bool)
    for cid in range(1, ncomp + 1):
        comp = (lbl == cid)
        if comp.sum() < MIN_SIZE:
            small |= comp

    active1_clean = active1_base & (~small)

    idomain = np.stack([active1_clean.astype(int), active2_base.astype(int)], axis=0)

    flopy.mf6.ModflowGwfdis(gwf, nlay=2, nrow=nrow, ncol=ncol,
                             delr=dem_tr.a, delc=abs(dem_tr.e),
                             top=dem, botm=[bot1, BOT2],
                             idomain=idomain)
    flopy.mf6.ModflowGwfic(gwf, strt=[h0, h0])
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=[1,0], k=[soil_k, rock_k], k33=[soil_k/10.0, rock_k/10.0])

    sy1 = np.where((eff_sd > 0) & np.isfinite(eff_sd), 0.10, 0.0)   # specific yield in L1
    ss1 = 1e-5  # 1/m, small specific storage in L1
    ss2 = 1e-6  # 1/m, smaller in L2

    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=[1, 0],                  # must match the intent of NPF icelltype
        sy=[sy1, np.zeros_like(sy1)],     # Sy only in L1
        ss=[ss1*np.ones_like(sy1), ss2*np.ones_like(sy1)],
        steady_state={0: False},
        transient={0: True},
    )

    # --- Recharge package (m/day) with multiplier; replace NaNs by 0 for MF6
    # ---------- Recharge zoning ----------
    rch_mult = np.ones_like(rech, dtype=float)

    # Elevation-based multipliers
    if rch_elev_bins and rch_elev_factors:
        bins = np.asarray(rch_elev_bins, dtype=float)
        facs = np.asarray(rch_elev_factors, dtype=float)
        assert len(bins) >= 2, "rch_elev_bins needs at least two edges"
        assert len(facs) == len(bins) - 1, "rch_elev_factors must have len(bins)-1 values"

        # idx ∈ {0..len(bins)}; we want 1..len(bins)-1 as inside intervals
        idx = np.digitize(dem, bins=bins, right=False)  # left-closed intervals
        band_fac = np.ones_like(rech, dtype=float)
        for k in range(1, len(bins)):
            band_fac[idx == k] = float(facs[k - 1])
        rch_mult *= band_fac

    # Soil-class multipliers
    if rch_soil_factors:
        band_fac = np.ones_like(rech, dtype=float)
        for cls, fac in rch_soil_factors.items():
            band_fac[(soil_class == int(cls)) & catch_mask] *= float(fac)
        rch_mult *= band_fac

    rech = rech * rch_mult
    rech[~catch_mask] = np.nan

    # (optional) write the multiplier field for debugging
    save_array_as_geotiff(
        rch_mult, os.path.join(cache_dir, "recharge_multiplier_for_viz.tif"), dem_tr, dem_crs
    )


    rech_scaled = rech * rch_multiplier
    print(f"[recharge] applying multiplier={rch_multiplier:.3f}; final m/day min/max = "
          f"{np.nanmin(rech_scaled):.2e}/{np.nanmax(rech_scaled):.2e}")
    rech_mf = np.nan_to_num(rech_scaled, nan=0.0)
    rch_spd = {
    0: 0.4 * rech_mf,   # 40% during the 30-day ramp
    1: 1.0 * rech_mf    # 100% for the remaining 335 days
    }
    flopy.mf6.ModflowGwfrcha(gwf, recharge=rch_spd)
    
    

    # 7) Coastal GHB
    print_time("Setting up coastal GHB")

    # Build a mask of coastal cells (True where a GHB should be applied)
    coast_mask = create_coastal_mask(
        filepaths['coast'], [catch_poly], coastal_buffer, dem.shape, dem_tr, dem_crs
    )

    # Save mask for visualization
    cm = coast_mask.astype(np.uint8)
    save_array_as_geotiff(
        cm,
        os.path.join(cache_dir, 'coastal_ghb_mask_for_viz.tif'),
        dem_tr,
        dem_crs
    )

    # --- Sea level stage for the given year (m a.s.l.) ---
    stage_value = 0.0  # fallback if file missing or no matching year
    if "sea_level" in filepaths and os.path.exists(filepaths["sea_level"]):
        try:
            # support both comma- and semicolon-delimited files
            sea_df = pd.read_csv(filepaths["sea_level"], sep='[;,]', engine='python')
            if {"Year", "AvgSeaLevel_cm"}.issubset(sea_df.columns):
                match = sea_df.loc[sea_df["Year"] == recharge_year, "AvgSeaLevel_cm"]
                if not match.empty:
                    # average all stations for that year (cm → m)
                    stage_value = float(match.mean()) / 100.0
            else:
                print("[GHB] Sea-level CSV missing 'Year' and/or 'AvgSeaLevel_cm' columns.")
        except Exception as e:
            print(f"[GHB] Sea-level read failed: {e}")
    print(f"[GHB] Using stage={stage_value:.3f} m a.s.l. for year {recharge_year}")

    # Build stress-period data: (cellid, stage, conductance)
    ghb_spd = []

    id1 = active1_clean.astype(int)        # L1 active
    id2 = active2_base.astype(int)         # L2 active (consistent with clamped bot1/BOT2)
    

    for i in range(nrow):
        for j in range(ncol):
            if not coast_mask[i, j]:
                continue

            # layer 1 bottom and layer 2 bottom
            
            bot1_ij = bot1[i, j]                   # clamped top of L2 / bottom of L1
            bot2 = ROCK_BOTTOM_ELEV                # bottom of L2 (constant)

            # decide which layer to place GHB on
            
            if id1[i, j] == 1 and (stage_value >= bot1_ij + 1e-3):
                # place on layer 0 (L1)
                lay = 0
                kcell = soil_k[i, j]
                thick = max(eff_sd[i, j], 0.5)
            elif id2[i, j] == 1 and (stage_value >= bot2 + 1e-3):
                # fall back to layer 1 (L2) if L1 is invalid
                lay = 1
                kcell = rock_k[i, j]
                thick = max((bot1_ij - ROCK_BOTTOM_ELEV), 1.0)
            else:
                # no valid receiving cell
                continue

            # conductance (cap to be safe)
            cond0 = kcell * cell_area / thick * ghb_cond_multiplier
            cond0 = min(cond0, 5e3) # was 5e4 before

            ghb_spd.append(((lay, i, j), stage_value, cond0))


    if ghb_spd:
        flopy.mf6.ModflowGwfghb(gwf, stress_period_data={0: ghb_spd}, maxbound=len(ghb_spd))
    else:
        print("[GHB] No coastal cells found; skipping GHB.")
    
    print(f"[diag] GHB cells: {len(ghb_spd) if ghb_spd else 0}")


    # 8) River RIV
    print_time("Setting up river package")
    rivers    = gpd.read_file(filepaths['rivers']).to_crs(dem_crs)
    riv_cells = rasterize(
        [(g, 1) for g in rivers.geometry],
        out_shape=(nrow, ncol),
        transform=dem_tr,
        fill=0,
        all_touched=False,  # was True
        dtype="uint8"
    )
    
    #rc[~catch_mask] = 0
    riv_cells[~catch_mask] = 0
    rc = riv_cells.astype(np.uint8)
    # --- River conductance parameters (simple defaults) ---
    cell_len   = max(dem_tr.a, abs(dem_tr.e))  # cell size [m]
    reach_width = 5.0                          # generic small river width [m]
    rb_thick    = 1.0                           # riverbed thickness [m]

    # Choose ONE of these Kriverbed approaches:
    use_relative_rbK = True     # True: fraction of aquifer K; False: constant value
    rbK_factor       = 0.01    # was 0.05 - 5% of aquifer K is common when no data
    rbK_const        = 1e-6 * 86400.0  # m/s -> m/day (~0.0864 m/day)

    save_array_as_geotiff(
        rc, os.path.join(cache_dir, 'river_cells_for_viz.tif'),
        dem_tr, 
        dem_crs,
    
    )
    
    riv_spd = []
    for i in range(nrow):
        for j in range(ncol):
            if not riv_cells[i, j]:
                continue

            stage = dem[i, j] - 1.5
            bot1_ij = bot1[i, j]
            bot2    = ROCK_BOTTOM_ELEV

            # choose layer: prefer L1 if it exists and river is above bot1; else use L2 if river is above bot2
            if (id1[i, j] == 1) and (stage > bot1_ij + 0.1) and (eff_sd[i, j] > 0):
                lay   = 0
                kcell = soil_k[i, j]
                rbot_candidate = max(stage - 1.0, bot1_ij + 0.05)
            elif (id2[i, j] == 1) and (stage > bot2 + 0.1):
                lay   = 1
                kcell = rock_k[i, j]
                # keep river bottom within L2 thickness
                rbot_candidate = max(stage - 1.0, bot2 + 0.05)
            else:
                continue
            rbot = min(rbot_candidate, stage - 0.05)     
            if use_relative_rbK:
                k_rb = kcell * rbK_factor
            else:
                k_rb = rbK_const       

            A_contact = reach_width * cell_len
            cond0 = (k_rb / rb_thick) * A_contact
            cond0 *= riv_cond_multiplier
            cond0 = min(cond0, 5e3)  
            riv_spd.append(((lay, i, j), stage, cond0, rbot)) 
                    
    if riv_spd:
        flopy.mf6.ModflowGwfriv(gwf, stress_period_data={0: riv_spd}, maxbound=len(riv_spd))

    print(f"[diag] RIV cells: {len(riv_spd) if riv_spd else 0}")
    

    

    
    

    has_riv = riv_cells.astype(bool)
    has_ghb = coast_mask
    active1 = (id1 == 1)

    # multi-cell-thick rim (2–3 cells inland) instead of just 1
    RIM_THICK = 1  # try 2 first; you can increase to 3 if needed
    # distance from inactive/edge (in pixels)
    dist = distance_transform_edt(active1)  # 0 at edge, grows inward
    rim_mask = (active1 & (dist <= RIM_THICK))

    # don't place where we already have GHB or RIV
    rim_mask &= (~has_ghb) & (~has_riv)

    rim_spd = []
    rim_factor = 0.2  # stronger leak than 0.2 (try 0.5; you can tweak 0.3–0.8)

    for i in range(nrow):
        for j in range(ncol):
            if not rim_mask[i, j]:
                continue
            if eff_sd[i, j] <= 0:
            
                continue
            land = dem[i, j]
            bot1_ij = bot1[i, j]
            # a touch deeper than before to grab perched heads
            elev = max(land - 1.0, bot1_ij + 0.05)
            cond = soil_k[i, j] * cell_area / max(eff_sd[i, j], 1.0) * rim_factor
            cond = min(cond, 1e4)  # allow more leak than before
            rim_spd.append(((0, i, j), elev, cond))

    
   
    if rim_spd:
        ModflowGwfdrn(gwf, stress_period_data={0: rim_spd}, maxbound=len(rim_spd))
    print(f"[diag] RIM-DRN cells: {len(rim_spd) if rim_spd else 0}")
        

    

    drn_spd = []
    drain_factor = 0.15  # was 0.3 before ↑ from 1.0; try 2.0–3.0 if heads still high
    
    has_ghb = coast_mask
    has_riv = riv_cells.astype(bool)

    for i in range(nrow):
        for j in range(ncol):
            if id1[i, j] != 1:
                continue  # only active soil-layer cells
            
            if has_riv[i, j] or has_ghb[i, j]:
                continue

            land_surf = dem[i, j]
            
            soil_thick = max(eff_sd[i, j], 0.0)

            # pull drains a bit LOWER to actively bleed highs
            desired_elev = land_surf - 0.8 # ↓ was 0.5
            bottom_l1 = bot1[i, j]
            elev_bot = bottom_l1 + 0.05
            elev = max(desired_elev, elev_bot)

            cond = soil_k[i, j] * cell_area / max(soil_thick, 0.5) * drain_factor
            cond = min(cond, 2e3) # was 5e4

            drn_spd.append(((0, i, j), elev, cond))

    if drn_spd:
        ModflowGwfdrn(gwf, stress_period_data={0: drn_spd}, maxbound=len(drn_spd))
    print(f"[diag] DRN cells: {len(drn_spd) if drn_spd else 0}")
  

        
    # 9) Output control & execution
    print_time("Writing and running simulation")
    flopy.mf6.ModflowGwfoc(gwf,
        head_filerecord=f"gwf_{catchment_id}.hds",
        budget_filerecord=f"gwf_{catchment_id}.cbc",
        saverecord=[('HEAD','ALL'), ('BUDGET','ALL')]
    )
    sim.write_simulation()
    success, _ = sim.run_simulation()

    # --- Early-abort: check IMS convergence log for divergence ---
    ims_csv = os.path.join(base_ws, 'ims.csv')
    if os.path.exists(ims_csv):
        try:
            ims_df = pd.read_csv(ims_csv)
            # Column 6 (0-indexed 5) is dvmax; column 5 (0-indexed 4) is inner_iteration
            cols = ims_df.columns.tolist()
            if len(cols) >= 6:
                dvmax_col = cols[5]
                dvmax_vals = pd.to_numeric(ims_df[dvmax_col], errors='coerce').abs()
                max_dvmax = dvmax_vals.max()
                last_dvmax = dvmax_vals.iloc[-1] if len(dvmax_vals) > 0 else 0
                print(f"[IMS] worst dvmax = {max_dvmax:.2e}, final dvmax = {last_dvmax:.2e}")
                if max_dvmax > 1e4:
                    print(f"[IMS] WARNING: dvmax reached {max_dvmax:.2e} m — model likely diverged")
        except Exception as e:
            print(f"[IMS] Could not parse ims.csv: {e}")

    if not success:
        # Check if results exist anyway (CONTINUE mode may have pushed through)
        hds_path = os.path.join(base_ws, f"gwf_{catchment_id}.hds")
        if os.path.exists(hds_path) and os.path.getsize(hds_path) > 0:
            print("[WARNING] MODFLOW reported convergence failure but produced output — continuing with results")
        else:
            print(f"[FAIL] Catchment {catchment_id}: MODFLOW run failed with no output")
            return None, None, None, catch_poly
    
    # --- quick water-budget summary (last time step) ---
    cbcfile = os.path.join(base_ws, f"gwf_{catchment_id}.cbc")
    try:
        cbc = flopy.utils.CellBudgetFile(cbcfile, precision='double')
        names = cbc.get_unique_record_names()
        totin = totout = 0.0
        for nm in names:
            key = nm.decode() if isinstance(nm, (bytes, bytearray)) else nm
            recs = cbc.get_data(text=key)
            if not recs:
                continue
            rec = recs[-1]  # last time step
            if isinstance(rec, np.ndarray):
                names = rec.dtype.names or ()
                qarr = rec['q'] if ('q' in names) else rec
                qsum = float(np.nansum(qarr))
            else:
                qsum = float(rec) if np.isscalar(rec) else float(np.nansum(rec))
            
            if qsum >= 0:
                totin += qsum
            else:
                totout += -qsum
        print(f"[budget] IN={totin:.3e} m³/d  OUT={totout:.3e} m³/d  | diff={totin - totout:.3e}")
    except Exception as e:
        print(f"[budget] Could not read budget: {e}")

    # 10) Read final heads
    print_time("Reading final heads")

    hfile = os.path.join(base_ws, f"gwf_{catchment_id}.hds")
    if not os.path.exists(hfile) or os.path.getsize(hfile) == 0:
        print(f"[FAIL] Catchment {catchment_id}: Head file is empty or missing")
        return None, None, None, catch_poly
    hf = flopy.utils.HeadFile(hfile)
    times = hf.get_times()
    heads3d = hf.get_data(totim=times[-1]).astype(float)  # shape (nlay, nrow, ncol)
    h1 = heads3d[0]
    h2 = heads3d[1]

    # Composite: prefer L1 if active; otherwise use L2 if active; else NaN
    head_comp = np.where(idomain[0] == 1, h1,
             np.where(idomain[1] == 1, h2, np.nan))

    # basic cleaning
    head_comp[~np.isfinite(head_comp)] = np.nan
    head_comp[np.abs(head_comp) > 1e6] = np.nan
    head_comp[~catch_mask] = np.nan

    # --- diagnose outliers relative to land surface & absolute ceiling ---
    bad1 = np.isfinite(head_comp) & (head_comp > dem + 150.0)   # >150 m above ground
    bad2 = np.isfinite(head_comp) & (head_comp > 5e3)           # >5000 m absolute
    bad  = bad1 | bad2
    print(f"[heads] bad1={int(bad1.sum())} cells (>DEM+150 m), bad2={int(bad2.sum())} cells (>5,000 m)")

    save_array_as_geotiff(bad.astype(np.uint8),  os.path.join(cache_dir, "head_outlier_mask.tif"),        dem_tr, dem_crs)
    save_array_as_geotiff(bad1.astype(np.uint8), os.path.join(cache_dir, "head_outlier_aboveDEM150.tif"), dem_tr, dem_crs)
    save_array_as_geotiff(bad2.astype(np.uint8), os.path.join(cache_dir, "head_outlier_gt5000.tif"),      dem_tr, dem_crs)

    head_viz = head_comp.copy()
    head_viz[bad] = np.nan

    finite = np.isfinite(head_viz)
    if finite.any():
        q2, q98 = np.nanpercentile(head_viz[finite], [2, 98])
        print(f"[heads] robust 2–98% range (viz): {q2:.2f}…{q98:.2f} m | max={np.nanmax(head_viz):.2f}")

        save_array_as_geotiff(
            head_viz,
            os.path.join(cache_dir, 'final_head_for_viz.tif'),
            dem_tr, dem_crs,
        )
    
    save_array_as_geotiff(idomain[0].astype('uint8'),
        os.path.join(cache_dir,'idomain_L1_for_viz.tif'), dem_tr, dem_crs)
    save_array_as_geotiff(idomain[1].astype('uint8'),
        os.path.join(cache_dir,'idomain_L2_for_viz.tif'), dem_tr, dem_crs)
    print_time("Finished setup_and_run_modflow")
    return head_comp, dem_tr, dem_crs, catch_poly

    
