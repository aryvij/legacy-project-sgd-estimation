#!/usr/bin/env python3
"""
Description:
------------
This script reads the SGU soil-depth raster (jorddjupsmodell_10x10m.tif),
resamples it to match the DEM’s resolution/extent (elevation_sweden.tif),
and writes out a new soil-depth raster aligned with the DEM grid.

Usage:
------
1. Adjust the file paths if needed.
2. From a terminal, run:
   python soil_depth_preprocess.py

Outputs:
--------
A GeoTIFF named "soil_depth_resampled.tif" (or similar), stored in
data/output/ (or wherever you choose), which you can then use
in your main groundwater modeling code.
"""

#!/usr/bin/env python3
"""
Description:
------------
This script reads the SGU soil-depth raster (jorddjupsmodell_10x10m.tif),
resamples it to match the DEM’s resolution/extent (elevation_sweden.tif),
and writes out a new soil-depth raster aligned with the DEM grid.

Paths:
------
Input soil-depth:  data/input/aquifer_data/jorddjupsmodell/jorddjupsmodell_10x10m.tif
DEM:               data/input/dem/elevation_sweden.tif
Output:            data/output/soil_depth_resampled.tif   <-- CHANGED

Usage:
------
From a terminal, run:

    python soil_depth_preprocess.py

Make sure this script is in e.g.:
   src/preprocessing/soil_depth_preprocess.py

and your folder structure is like:

  SGD_model/
  ├─ data/
  │  ├─ input/
  │  │  ├─ aquifer_data/
  │  │  │   └─ jorddjupsmodell/
  │  │  │       └─ jorddjupsmodell_10x10m.tif
  │  │  ├─ dem/
  │  │  │   └─ elevation_sweden.tif
  │  │  ...
  │  ├─ output/  <-- must exist or will be created
  └─ src/
     └─ preprocessing/
         └─ soil_depth_preprocess.py

Outputs:
--------
soil_depth_resampled.tif in data/output

"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def main():
    # -------------------------------------------
    # 1) Define paths relative to this script
    # -------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))

    soil_depth_path = os.path.join(
        script_dir,
        '..', '..', 
        'data', 'input', 'aquifer_data', 'jorddjupsmodell',
        'jorddjupsmodell_10x10m.tif'
    )

    dem_path = os.path.join(
        script_dir,
        '..', '..', 
        'data', 'input', 'dem',
        'elevation_sweden.tif'
    )

    # Output directory changed to data/output
    output_dir = os.path.join(script_dir, '..', '..', 'data', 'output')
    os.makedirs(output_dir, exist_ok=True)

    resampled_soil_depth_path = os.path.join(output_dir, 'soil_depth_resampled.tif')

    print("Soil Depth raster:", soil_depth_path)
    print("DEM raster:", dem_path)
    print("Output will be saved to:", resampled_soil_depth_path)

    # -------------------------------------------
    # 2) Read DEM info
    # -------------------------------------------
    with rasterio.open(dem_path) as dem_src:
        dem_data = dem_src.read(1)
        dem_profile = dem_src.profile
        dem_transform = dem_profile['transform']
        dem_crs = dem_profile['crs']
        dem_width = dem_profile['width']
        dem_height = dem_profile['height']

    print(f"DEM shape: {dem_height} x {dem_width}")
    print("DEM transform:", dem_transform)
    print("DEM CRS:", dem_crs)

    # -------------------------------------------
    # 3) Read soil-depth raster
    # -------------------------------------------
    with rasterio.open(soil_depth_path) as soil_src:
        soil_depth_data = soil_src.read(1)
        soil_transform = soil_src.transform
        soil_crs = soil_src.crs
        soil_profile = soil_src.profile

    print(f"Soil-depth shape: {soil_profile['height']} x {soil_profile['width']}")
    print("Soil-depth transform:", soil_transform)
    print("Soil-depth CRS:", soil_crs)

    # -------------------------------------------
    # 4) Prepare array for resampled data
    # -------------------------------------------
    resampled_soil = np.zeros((dem_height, dem_width), dtype=np.float32)

    # -------------------------------------------
    # 5) Reproject soil-depth -> DEM grid
    # -------------------------------------------
    reproject(
        source=soil_depth_data,
        destination=resampled_soil,
        src_transform=soil_transform,
        src_crs=soil_crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=Resampling.bilinear
    )

    # Optional: set negative or nonsensical values to nodata
    # We use a default or from soil_profile
    no_data_val = soil_profile.get('nodata', -9999.0)
    resampled_soil[resampled_soil < 0] = no_data_val

    # -------------------------------------------
    # 6) Build new profile
    # -------------------------------------------
    out_profile = dem_profile.copy()
    out_profile.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': no_data_val
    })

    # -------------------------------------------
    # 7) Save resampled soil-depth
    # -------------------------------------------
    with rasterio.open(resampled_soil_depth_path, 'w', **out_profile) as dst:
        dst.write(resampled_soil, 1)

    print("Resampled soil depth saved to:", resampled_soil_depth_path)
    print("Done.")

if __name__ == "__main__":
    main()
