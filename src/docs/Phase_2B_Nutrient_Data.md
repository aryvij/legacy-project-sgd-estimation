# Phase 2B: Nutrient Data & Concentration Mapping

> **Goal:** Obtain, process, and spatially map groundwater nitrogen (N) and phosphorus (P) concentrations for coastal catchments. These concentrations will be used in Phase 2C to compute nutrient loads to the coast.

---

## Overview

To estimate nutrient loads delivered by submarine groundwater discharge, you need spatially distributed groundwater nutrient concentrations. Sweden's primary source is the SGU (Geological Survey of Sweden) groundwater chemistry database, which contains lab-analyzed samples from wells across the country.

---

## What You Need

### Data

| Item | Description | Where to Get It | Format |
|------|-------------|-----------------|--------|
| **SGU groundwater chemistry** | N and P concentrations from monitored wells | SGU — see download instructions below | GeoPackage / CSV |
| **Catchment boundaries** | Already available from Phase 1 | `data/input/shapefiles/catchment/bsdbs.shp` | Shapefile |
| **Well locations** | Already available from Phase 1 | `data/input/well_data/brunnar.gpkg` | GeoPackage |
| **DEM / model grid** | For spatial mapping | Phase 1 cached outputs | GeoTIFF |
| **Soil permeability classes** | For zone-based concentration assignment | `data/input/aquifer_data/genomslapplighet/genomslapplighet.gpkg` | GeoPackage |

### Software

- Python packages: `geopandas`, `pandas`, `numpy`, `rasterio`, `scipy`, `matplotlib`
- Optional: `scikit-learn` (for kriging or ML-based interpolation)

---

## Step-by-Step Instructions

### Task 2B-1: Download Groundwater Nutrient Data from SGU

**SGU provides groundwater chemistry data through several channels:**

#### Option A — SGU Kartvisare (Map Viewer) + Data Export

1. Go to [SGU Kartvisare](https://apps.sgu.se/kartvisare/kartvisare-grundvattenkemi.html) (Grundvattenkemi = Groundwater Chemistry).
2. Zoom to your study area (coastal catchments along the Swedish coast).
3. Click on wells to view available parameters. Look for:
   - **Nitrat (NO₃)** or **Kväve, totalt (total nitrogen)**
   - **Fosfat (PO₄)** or **Fosfor, totalt (total phosphorus)**
4. Use the export/download feature to download data for selected wells.

#### Option B — SGU Open Data API

1. Check [SGU's open data portal](https://www.sgu.se/produkter-och-tjanster/geologiska-data/oppna-data/) for downloadable datasets.
2. Look for datasets labelled "Grundvattenkemi" or "Groundwater chemistry".
3. Data is typically in GeoPackage or CSV format with coordinates in SWEREF99 TM (EPSG:3006).

#### Option C — Request from SGU

If the data is not freely available in bulk:
1. Contact SGU at kundservice@sgu.se
2. Request: "Groundwater chemistry data (nitrogen and phosphorus species) for wells in coastal catchments of [your study area], in digital format with coordinates."
3. Specify that you need: well ID, coordinates, sampling date, parameter name, value, unit.

#### What to Look For in the Data

The most common nutrient parameters in Swedish groundwater monitoring:

| Parameter | Swedish Name | Unit | Typical Range |
|-----------|-------------|------|---------------|
| Nitrate | Nitrat (NO₃⁻) | mg/L | 0–50 |
| Total nitrogen | Kväve, totalt | mg/L | 0–20 |
| Ammonium | Ammonium (NH₄⁺) | mg/L | 0–5 |
| Phosphate | Fosfat (PO₄³⁻) | mg/L | 0–1 |
| Total phosphorus | Fosfor, totalt | mg/L | 0–0.5 |

**Priority:** Total nitrogen and total phosphorus are the most useful for load calculations.

#### Suggested File Location

Save downloaded data to:
```
data/input/nutrient_data/
├── groundwater_chemistry_sgu.gpkg    (or .csv)
└── README_nutrient_data_source.txt   (document data source, date downloaded, any filters)
```

---

### Task 2B-2: Process Nutrient Concentrations

**Objective:** Clean, filter, and prepare the nutrient data for spatial mapping.

**Suggested file:** `src/core/nutrient_processing.py`

#### Step 1: Load and Inspect

```python
import geopandas as gpd
import pandas as pd

# Load the data
chem = gpd.read_file("data/input/nutrient_data/groundwater_chemistry_sgu.gpkg")

# Inspect columns
print(chem.columns.tolist())
print(chem.head())
print(f"Total records: {len(chem)}")
print(f"Unique wells: {chem['well_id'].nunique()}")  # adjust column name
```

#### Step 2: Filter to Study Area

```python
# Load catchment boundaries
catchments = gpd.read_file("data/input/shapefiles/catchment/bsdbs.shp")

# Reproject if needed
if chem.crs != catchments.crs:
    chem = chem.to_crs(catchments.crs)

# Spatial join: keep only wells inside study catchments
chem_in_area = gpd.sjoin(chem, catchments, how="inner", predicate="within")
print(f"Records in study area: {len(chem_in_area)}")
```

#### Step 3: Extract N and P Values

```python
# Map parameter names to standardized columns
# Adjust these based on actual column names in the SGU data
N_PARAMS = ['nitrat', 'kväve_totalt', 'NO3', 'N_tot']
P_PARAMS = ['fosfat', 'fosfor_totalt', 'PO4', 'P_tot']

# Filter and pivot (if data is in long format)
n_data = chem_in_area[chem_in_area['parameter'].str.lower().isin([p.lower() for p in N_PARAMS])]
p_data = chem_in_area[chem_in_area['parameter'].str.lower().isin([p.lower() for p in P_PARAMS])]
```

#### Step 4: Quality Control

```python
# Remove negative values
n_data = n_data[n_data['value'] >= 0]
p_data = p_data[p_data['value'] >= 0]

# Remove extreme outliers (> 99th percentile or physically unreasonable)
n_data = n_data[n_data['value'] < 100]  # mg/L cap for N
p_data = p_data[p_data['value'] < 5]    # mg/L cap for P

# If multiple samples per well, compute mean (or median)
n_mean = n_data.groupby('well_id').agg({
    'value': 'median',
    'geometry': 'first'
}).reset_index()

p_mean = p_data.groupby('well_id').agg({
    'value': 'median',
    'geometry': 'first'
}).reset_index()
```

#### Step 5: Unit Conversion

For load calculations in Phase 2C, you need **kg/m³**:
```python
# mg/L = g/m³, so divide by 1000 to get kg/m³
n_mean['conc_kg_m3'] = n_mean['value'] / 1000.0
p_mean['conc_kg_m3'] = p_mean['value'] / 1000.0
```

#### Step 6: Save Processed Data

```python
n_mean_gdf = gpd.GeoDataFrame(n_mean, crs=catchments.crs)
p_mean_gdf = gpd.GeoDataFrame(p_mean, crs=catchments.crs)

n_mean_gdf.to_file("data/input/nutrient_data/nitrogen_processed.gpkg", driver="GPKG")
p_mean_gdf.to_file("data/input/nutrient_data/phosphorus_processed.gpkg", driver="GPKG")
```

---

### Task 2B-3: Map Nutrient Concentrations to the Model Grid

**Objective:** Create a 2D raster of N and P concentrations on the same grid as the MODFLOW model.

There are two approaches depending on data density:

#### Approach A — Spatial Interpolation (if many wells)

If you have ≥10–15 wells with nutrient data in the catchment:

```python
from scipy.interpolate import griddata
import numpy as np
import rasterio

# Load the model grid info from a cached DEM
with rasterio.open(f"data/output/cache/{catchment_id}/dem_clipped.tif") as src:
    dem = src.read(1)
    transform = src.transform
    crs = src.crs

nrow, ncol = dem.shape

# Build grid coordinates (same as in sgd_utils.py)
xs = np.linspace(transform[2] + transform[0]/2,
                 transform[2] + transform[0]*(ncol - 0.5), ncol)
ys = np.linspace(transform[5] + transform[4]/2,
                 transform[5] + transform[4]*(nrow - 0.5), nrow)
grid_x, grid_y = np.meshgrid(xs, ys)

# Well coordinates and values
pts = np.column_stack([n_mean_gdf.geometry.x, n_mean_gdf.geometry.y])
vals = n_mean_gdf['conc_kg_m3'].values

# Interpolate (linear + nearest fill)
conc_grid = griddata(pts, vals, (grid_x, grid_y), method='linear')
nan_mask = np.isnan(conc_grid)
if nan_mask.any():
    conc_grid[nan_mask] = griddata(pts, vals, (grid_x, grid_y), method='nearest')[nan_mask]
```

#### Approach B — Zone-Based Assignment (if few wells)

If you have sparse data, assign concentrations by zone (e.g., soil permeability class or land use):

```python
# Example: assign based on soil permeability class
# Class 1 (low perm) → lower N due to slower transport, more denitrification
# Class 3 (high perm) → higher N due to faster leaching
N_BY_CLASS = {1: 0.002, 2: 0.005, 3: 0.010}  # kg/m³

with rasterio.open(f"data/output/cache/{catchment_id}/soil_class.tif") as src:
    soil_class = src.read(1)

n_conc = np.zeros_like(dem, dtype=float)
for cls, conc in N_BY_CLASS.items():
    n_conc[soil_class == cls] = conc
```

#### Save Concentration Rasters

```python
from core.sgd_utils import save_array_as_geotiff

save_array_as_geotiff(n_conc, f"data/output/cache/{catchment_id}/nitrogen_conc_kg_m3.tif",
                      transform, crs, unit_name="kg/m3")
save_array_as_geotiff(p_conc, f"data/output/cache/{catchment_id}/phosphorus_conc_kg_m3.tif",
                      transform, crs, unit_name="kg/m3")
```

---

## Data Gaps & Documentation

Record the following in your output:

1. **Spatial coverage:** How many wells with nutrient data exist in each catchment?
2. **Temporal coverage:** What date range do the samples cover? Are they representative of current conditions?
3. **Data density:** Is interpolation justified, or must you fall back to zone-based assignment?
4. **Below-detection-limit values:** How are non-detects handled? (Common approach: use half the detection limit)
5. **N species:** Are you using NO₃, total N, or another form? Document the choice and conversion.

---

## Expected Output Files

```
data/input/nutrient_data/
├── groundwater_chemistry_sgu.gpkg            ← raw download
├── nitrogen_processed.gpkg                    ← cleaned, filtered, with conc_kg_m3
├── phosphorus_processed.gpkg                  ← cleaned, filtered, with conc_kg_m3
└── README_nutrient_data_source.txt            ← data provenance

data/output/cache/<catchment_id>/
├── nitrogen_conc_kg_m3.tif                    ← gridded N concentration
└── phosphorus_conc_kg_m3.tif                  ← gridded P concentration
```

---

## Verification Checklist

- [ ] Nutrient data downloaded and saved with provenance documentation
- [ ] Data filtered to study catchments
- [ ] QC applied: no negative values, outliers removed, non-detects handled
- [ ] Unit conversion to kg/m³ verified (mg/L ÷ 1000)
- [ ] Concentration raster on the same grid as model DEM (size, CRS, transform match)
- [ ] Concentration values are physically reasonable:
  - N: typically 0.001–0.020 kg/m³ (1–20 mg/L)
  - P: typically 0.00005–0.0005 kg/m³ (0.05–0.5 mg/L)
- [ ] Data gaps and assumptions documented

---

## Key References

- SGU Kartvisare (groundwater chemistry): https://apps.sgu.se/kartvisare/kartvisare-grundvattenkemi.html
- SGU Open Data: https://www.sgu.se/produkter-och-tjanster/geologiska-data/oppna-data/
- Swedish drinking water standards (nitrate limit 50 mg/L): Livsmedelsverket
- Typical Swedish groundwater nutrient levels: Naturvårdsverket rapport series
