# SGD Estimation — MODFLOW-6 Groundwater Model

> **Submarine Groundwater Discharge (SGD) and coastal nutrient loading estimation using MODFLOW-6**  
> Developed at KTH Royal Institute of Technology

---

## Quick Start

### Prerequisites

- **Python 3.10+** with the following packages:
  ```
  flopy rasterio geopandas scipy numpy pandas matplotlib shapely
  ```
  Optional: `SALib` (sensitivity analysis), `streamlit` (web UI)

- **MODFLOW 6** executable — download from [USGS](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model).  
  Pass the path via `--mf6 "path/to/mf6.exe"`.

### Run a simulation

```powershell
cd scripts
python src/main_sgd.py --catchment 204 --year 2010 --mf6 "path/to/mf6.exe"
```

This will:
1. Check if catchment 204 is coastal (if inland, exits — no SGD)
2. Build a two-layer MODFLOW-6 model (soil + fractured bedrock)
3. Run the simulation and save head maps + budget files

---

## Repository Structure

```
scripts/
├── README.md              ← you are here
└── src/
    ├── __init__.py
    │
    ├── CORE ──────────────────────────────────────────────────
    ├── main_sgd.py             ← CLI entry point
    ├── modflow_setup.py        ← builds & runs MODFLOW-6 model
    ├── flow_estimator.py       ← discharge lookup (monitored Q or regression)
    ├── sgd_utils.py            ← shared utilities (raster I/O, well interpolation, masks)
    │
    ├── CALIBRATION & VALIDATION ──────────────────────────────
    ├── calibration_with_figures.py  ← 5-multiplier calibration (scipy.optimize + plots)
    ├── validation.py                ← forward-run on unseen years
    │
    ├── DIAGNOSTICS ───────────────────────────────────────────
    ├── diagnostics.py          ← RMSE, scatter, QQ, stratified residuals
    ├── analyze_residuals.py    ← residual binning by elevation/distance
    ├── sgd_post.py             ← extract SGD (m³/day) from MODFLOW budget
    │
    ├── SENSITIVITY & UNCERTAINTY ─────────────────────────────
    ├── sensitivity_oat.py      ← one-at-a-time perturbation analysis
    ├── sensitivity_sobol.py    ← global Sobol sensitivity (SALib)
    ├── uncertainty_mc.py       ← Monte Carlo parameter ensemble
    │
    ├── PLOTTING ──────────────────────────────────────────────
    ├── plot_oat_results.py        ← OAT bar/line plots
    ├── plot_sobol_dual.py         ← dual-panel Sobol index charts
    ├── plot_uncertainty_violin.py ← MC ensemble violin plots
    │
    ├── UI ────────────────────────────────────────────────────
    ├── interface_main_sgd.py   ← Streamlit web interface
    │
    ├── DOCS ──────────────────────────────────────────────────
    ├── Calibration of MODFLOW-6 Model for SGD.md
    │
    ├── Preprocessing/          ← one-time data preparation
    │   ├── clipping_coast_line.py
    │   ├── discharge_data_mean_calculation.py
    │   ├── soil_dept_data_preprocess.py
    │   └── year_selection.py
    │
    └── visualisation/          ← standalone visualization helpers
        ├── calibration_2010_rmse_fromcsv.py
        ├── check_masks.py
        ├── ghb_riv_flux_SWIM.py
        ├── gridsize.py
        ├── Grid_visualisation_3D.py
        ├── grid_visualisation_SWIM.py
        ├── make_thumbnails.py
        ├── plot_results.py
        ├── riv_ghb_png.py
        └── soil_head_3d.py
```

---

## How the Model Works

### Two-Layer MODFLOW-6 Grid

| Layer | Material | Type | Top | Bottom |
|-------|----------|------|-----|--------|
| 1 | Soil (quaternary deposits) | Convertible | DEM surface | DEM − soil_depth (clamped ≥ −49.9 m) |
| 2 | Fractured bedrock | Confined | L1 bottom | −50 m (constant) |

### Boundary Conditions

- **RCH** — spatially distributed recharge (annual GeoTIFF, mm/yr → m/day)
- **GHB** — coastal cells within a buffer (sea-level stage from SMHI)
- **RIV** — river cells rasterized from national river shapefile
- **DRN** — rim drains (catchment edge) + interior drains (prevent unrealistic head build-up)

### Solver
Newton with under-relaxation, BICGSTAB linear acceleration, 500 outer / 1000 inner iterations.

### Stress Periods
Pseudo-transient: SP1 = 30-day ramp (40% recharge), SP2 = 335-day run (100% recharge).

---

## Calibration

Five multipliers are optimized (Nelder-Mead, RMSE objective against observed well heads):

| Parameter | Description | Calibrated (Catchment 204, 2010) |
|-----------|-------------|----------------------------------|
| `soilK_multiplier` | Soil hydraulic conductivity | 1.4529 |
| `rockK_multiplier` | Bedrock hydraulic conductivity | 2.9112 |
| `riv_cond_multiplier` | River-bed conductance | 1.0624 |
| `ghb_cond_multiplier` | Coastal GHB conductance | 0.8435 |
| `rch_multiplier` | Spatial recharge | 0.9069 |

Additional spatial zoning:
- **Elevation-based recharge:** bins [0, 10, 30, 60, 200] m → factors [1.02, 1.1, 1.16, 1.20]
- **Soil-class K factors:** {1: 1.40, 2: 0.80, 3: 0.95}
- **Soil-class recharge factors:** {1: 0.80, 2: 1.10, 3: 1.05}

```powershell
python src/calibration_with_figures.py --catchment 204 --year 2010 --mf6 "path/to/mf6.exe" --maxiter 50
```

---

## SGD Extraction

SGD is extracted as **GHB outflows** (negative q in the cell-budget file):

```powershell
python src/sgd_post.py --cbc "data/output/model_runs/mf6_204/gwf_204.cbc"
```

---

## Input Data

All input data lives outside this repository under `data/` (not version-controlled due to size):

| Dataset | Format | Source | Path |
|---------|--------|--------|------|
| DEM | GeoTIFF | Lantmäteriet | `data/input/dem/elevation_sweden.tif` |
| Catchments | Shapefile | — | `data/input/shapefiles/catchment/bsdbs.shp` |
| Coastline | Shapefile | — | `data/input/shapefiles/coast_line/coastline.shp` |
| Soil permeability | GeoPackage | SGU | `data/input/aquifer_data/genomslapplighet/genomslapplighet.gpkg` |
| Soil depth | GeoTIFF | SGU | `data/input/aquifer_data/jorddjupsmodell/jorddjupsmodell_10x10m.tif` |
| Bedrock K | GeoTIFF | — | `data/input/other_rasters/hydraulic_conductivity.tif` |
| Wells | GeoPackage | SGU | `data/input/well_data/brunnar.gpkg` |
| Recharge (yearly) | GeoTIFF | EGDI/GLDAS | `data/output/recharge_yearly/recharge_egdi_gldas_<YEAR>.tif` |
| Sea level | CSV | SMHI | `data/input/sea_level/yearly_average_sea_level.csv` |
| Rivers | Shapefile | — | `data/input/shapefiles/surface_water/Surface_water/hl_riks.shp` |
| Lakes | Shapefile | — | `data/input/shapefiles/surface_water/scandinavian_waters_polygons/` |
| Discharge | CSV | SMHI | `data/input/discharge/monitored_mean_Q.csv` |

**CRS:** SWEREF99 TM (EPSG:3006) — all inputs are reprojected to match the DEM.

---

## Key Technical Notes

1. **Caching:** Expensive raster ops are cached under `data/output/cache/<catchment_id>/`. Delete the cache folder if input data changes. Set `FORCE_RECHARGE_REBUILD=1` to force recharge recalculation.

2. **MODFLOW 6 path:** Update the `--mf6` argument or change the default in `main_sgd.py` line 63.

3. **Layer geometry:** L1 bottom is clamped so it never goes below −49.9 m (L2 bottom is −50 m constant). The Newton solver handles wetting/drying in L1.

4. **Soil permeability classes:** SGU genomsläpplighet classes 1–3 mapped to K: {1: 1e-8, 2: 1e-6, 3: 1e-5} m/s (converted to m/day internally).

---

## Command Reference

| Task | Command |
|------|---------|
| **Simulation** | `python src/main_sgd.py --catchment 204 --year 2010 --mf6 "path/to/mf6.exe"` |
| **Calibration** | `python src/calibration_with_figures.py --catchment 204 --year 2010 --mf6 "..." --maxiter 50` |
| **Validation** | `python src/validation.py --catchment 204 --calib-year 2010 --years 2018 2019 --mf6 "..."` |
| **Sensitivity (OAT)** | `python src/sensitivity_oat.py --catchment 204 --year 2010 --mf6 "..."` |
| **Sensitivity (Sobol)** | `python src/sensitivity_sobol.py --catchment 204 --year 2010 --mf6 "..."` |
| **Uncertainty (MC)** | `python src/uncertainty_mc.py --catchment 204 --year 2010 --mf6 "..."` |
| **SGD extraction** | `python src/sgd_post.py --cbc "data/output/model_runs/mf6_204/gwf_204.cbc"` |
| **Web UI** | `streamlit run src/interface_main_sgd.py` |

---

## Phase 2 — Upcoming Work

The following tasks build upon the existing SGD estimation framework:

### 2A: Time-Varying Recharge & Transient SGD
- Apply monthly scaling factors to annual recharge → 12 stress periods/year
- Modify `modflow_setup.py` TDIS for monthly periods
- Extract monthly SGD time series from budget files

### 2B: Nutrient Data & Concentration Mapping
- Download groundwater N and P data from the SGU database
- Process, QC, and spatially interpolate nutrient concentrations
- Create concentration maps for coastal catchments

### 2C: Coastal Nutrient Load Estimation
- Compute loads: `Load [kg/day] = SGD [m³/day] × Concentration [kg/m³]`
- Propagate uncertainty from both SGD and concentration estimates

### 2D (Optional): Groundwater Transport (MODFLOW 6 GWT)
- Couple GWT with the existing GWF model
- Calibrate transport parameters (dispersivity, porosity) against well observations

See `AGENTS.md` in the project root for detailed task descriptions.

---

## License

Internal project — KTH Royal Institute of Technology.

## Contact

- **Phase 1 author:** Arya Vijayan (aryathriveni@gmail.com)
