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

### Run a single catchment

All scripts must be run from the `scripts/` directory using `python -m`:

```powershell
cd scripts
python -m src.core.main_sgd --catchment 204 --year 2010 --mf6 "path/to/mf6.exe"
```

This will:
1. Check if catchment 204 is coastal (if inland, exits — no SGD)
2. Build a two-layer MODFLOW-6 model (soil + fractured bedrock)
3. Run the simulation and save head maps + budget files

### Run all coastal catchments (batch mode)

To run every Swedish coastal catchment in a single batch:

```powershell
python -m src.core.main_sgd --catchments all --year 2010 --mf6 "path/to/mf6.exe" `
  --data-root "C:\path\to\data\input" `
  --output-dir "C:\path\to\data\output"
```

Or specify a comma-separated list:

```powershell
python -m src.core.main_sgd --catchments 204,301,415 --year 2010 --mf6 "path/to/mf6.exe"
```

**Batch options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--catchments all` | Run all Swedish coastal catchments | — |
| `--catchments 204,301` | Run specific catchments | — |
| `--max-area 5000` | Skip catchments larger than this (km²) | 5000 |
| `--cell-size 200` | Resample DEM to this cell size (m) | native resolution |

### Resume support

The batch runner saves progress to `<output-dir>/batch_results_<YEAR>.xlsx` after **every** catchment. If the run is interrupted (crash, restart, etc.), simply re-run the same command — it will:
- Load the existing results file
- Skip all previously successful catchments
- Re-run only remaining and previously failed catchments

You will see: `Resuming: N catchments already done — skipping them`

> **Note:** If `batch_results_<YEAR>.xlsx` is open in Excel, the script cannot update it and will exit with an error. Close it before running.

### Custom data paths

By default, scripts look for input data at `scripts/data/input/` and write outputs to `scripts/data/output/`. If your data lives elsewhere, use `--data-root` and `--output-dir`:

```powershell
python -m src.core.main_sgd --catchment 204 --year 2010 --mf6 "path/to/mf6.exe" `
  --data-root "C:\path\to\data\input" `
  --output-dir "C:\path\to\data\output"
```

These two flags are available on **all** scripts (simulation, calibration, validation, sensitivity, uncertainty, diagnostics). When omitted, paths default to `<project>/data/input` and `<project>/data/output` relative to the repository.

---

## Repository Structure

```
scripts/
├── README.md              ← you are here
├── .gitignore
└── src/
    ├── __init__.py
    │
    ├── core/                       ← CORE MODULES
    │   ├── __init__.py
    │   ├── main_sgd.py             ← CLI entry point
    │   ├── modflow_setup.py        ← builds & runs MODFLOW-6 model
    │   ├── flow_estimator.py       ← discharge lookup (monitored Q or regression)
    │   └── sgd_utils.py            ← shared utilities (raster I/O, well interpolation, masks)
    │
    ├── calibration/                ← CALIBRATION & VALIDATION
    │   ├── __init__.py
    │   ├── calibration_with_figures.py  ← 5-multiplier calibration (scipy.optimize + plots)
    │   └── validation.py               ← forward-run on unseen years
    │
    ├── diagnostics/                ← DIAGNOSTICS
    │   ├── __init__.py
    │   ├── diagnostics.py          ← RMSE, scatter, QQ, stratified residuals
    │   ├── analyze_residuals.py    ← residual binning by elevation/distance
    │   └── sgd_post.py             ← extract SGD (m³/day) from MODFLOW budget
    │
    ├── sensitivity/                ← SENSITIVITY & UNCERTAINTY
    │   ├── __init__.py
    │   ├── sensitivity_oat.py      ← one-at-a-time perturbation analysis
    │   ├── sensitivity_sobol.py    ← global Sobol sensitivity (SALib)
    │   └── uncertainty_mc.py       ← Monte Carlo parameter ensemble
    │
    ├── plotting/                   ← POST-PROCESSING PLOTS
    │   ├── __init__.py
    │   ├── plot_oat_results.py        ← OAT bar/line plots
    │   ├── plot_sobol_dual.py         ← dual-panel Sobol index charts
    │   └── plot_uncertainty_violin.py ← MC ensemble violin plots
    │
    ├── interface_main_sgd.py       ← Streamlit web interface
    │
    ├── docs/                       ← DOCUMENTATION
    │   ├── Calibration of MODFLOW-6 Model for SGD.md
    │   ├── Phase_2A_Transient_Recharge_SGD.md
    │   ├── Phase_2B_Nutrient_Data.md
    │   ├── Phase_2C_Nutrient_Loads.md
    │   └── Phase_2D_GWT_Transport.md
    │
    ├── Preprocessing/              ← one-time data preparation
    │   ├── clipping_coast_line.py
    │   ├── discharge_data_mean_calculation.py
    │   ├── soil_dept_data_preprocess.py
    │   └── year_selection.py
    │
    └── visualisation/              ← standalone visualization helpers
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
Newton with under-relaxation (DBD), BICGSTAB linear acceleration, 200 outer / 500 inner iterations. Key convergence settings:
- **Complexity:** MODERATE (enables internal damping heuristics)
- **Under-relaxation theta:** 0.5 (applies 50% of each correction to prevent oscillation)
- **Backtracking:** 20 steps (rolls back overshooting corrections)
- **Convergence check:** after the run, `ims.csv` is parsed and a warning is printed if dvmax exceeds 10,000 m (indicates divergence)

### Initial Heads
Starting heads are set to DEM − 5 m (or from well interpolation where data exists). Coastal cells below 5 m elevation are clamped to at least sea level. This tight clamping helps Newton convergence — a poor initial guess can cause the solver to diverge.

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
python -m src.calibration.calibration_with_figures --catchment 204 --year 2010 --mf6 "path/to/mf6.exe" --maxiter 50 `
  --data-root "path/to/data/input" --output-dir "path/to/data/output"
```

---

## SGD Extraction

SGD is extracted as **GHB outflows** (negative q in the cell-budget file):

```powershell
python -m src.diagnostics.sgd_post --cbc "data/output/model_runs/mf6_204/gwf_204.cbc"
```

---

## Input Data

Input data lives outside this repository (not version-controlled due to size). Point scripts to it with `--data-root` and `--output-dir`. Paths below are relative to the data root:

| Dataset | Format | Source | Relative Path |
|---------|--------|--------|---------------|
| DEM | GeoTIFF | Lantmäteriet | `dem/elevation_sweden.tif` |
| Catchments | Shapefile | — | `shapefiles/catchment/bsdbs.shp` |
| Coastline | Shapefile | — | `shapefiles/coast_line/coastline.shp` |
| Coastal boundary check | Shapefile | — | `shapefiles/coastline_check/coastal_boundary.shp` |
| Soil permeability | GeoPackage | SGU | `aquifer_data/genomslapplighet/genomslapplighet.gpkg` |
| Soil depth | GeoTIFF | SGU | `aquifer_data/jorddjupsmodell/jorddjupsmodell_10x10m.tif` |
| Bedrock K | GeoTIFF | — | `other_rasters/hydraulic_conductivity.tif` |
| Wells | GeoPackage | SGU | `well_data/brunnar.gpkg` |
| Sea level | CSV | SMHI | `sea_level/yearly_average_sea_level.csv` |
| Rivers | Shapefile | — | `shapefiles/surface_water/Surface_water/hl_riks.shp` |
| Lakes | Shapefile | — | `shapefiles/surface_water/scandinavian_waters_polygons/` |
| Discharge | CSV | SMHI | `discharge/monitored_mean_Q.csv` |
| Recharge (yearly) | GeoTIFF | EGDI/GLDAS | *(output-dir)*: `recharge_yearly/recharge_egdi_gldas_<YEAR>.tif` |

**CRS:** SWEREF99 TM (EPSG:3006) — all inputs are reprojected to match the DEM.

---

## Key Technical Notes

1. **Data paths:** All scripts accept `--data-root` (input data folder) and `--output-dir` (output folder). When omitted, paths default to `scripts/data/input` and `scripts/data/output` relative to the project root.

2. **Caching:** Expensive raster ops are cached under `<output-dir>/cache/<catchment_id>/`. Delete the cache folder if input data changes. Set `FORCE_RECHARGE_REBUILD=1` to force recharge recalculation.

3. **MODFLOW 6 path:** Pass the path via `--mf6 "path/to/mf6.exe"`. Download from [USGS](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model).

4. **Layer geometry:** L1 bottom is clamped so it never goes below −49.9 m (L2 bottom is −50 m constant). The Newton solver handles wetting/drying in L1.

5. **Soil permeability classes:** SGU genomsläpplighet classes 1–3 mapped to K: {1: 1e-8, 2: 1e-6, 3: 1e-5} m/s (converted to m/day internally).

6. **Running scripts:** Always run from the `scripts/` directory using `python -m src.<module>`. The `src/__init__.py` sets up `sys.path` so that internal `from core.…` imports resolve correctly.

---

## Command Reference

All commands below should be run from the `scripts/` directory. Add `--data-root` and `--output-dir` to any command if your data is not at the default `scripts/data/` location.

| Task | Command |
|------|---------|
| **Single catchment** | `python -m src.core.main_sgd --catchment 204 --year 2010 --mf6 "path/to/mf6.exe"` |
| **Batch (all coastal)** | `python -m src.core.main_sgd --catchments all --year 2010 --mf6 "..." --data-root "..." --output-dir "..."` |
| **Batch (specific)** | `python -m src.core.main_sgd --catchments 204,301,415 --year 2010 --mf6 "..."` |
| **Calibration** | `python -m src.calibration.calibration_with_figures --catchment 204 --year 2010 --mf6 "..." --maxiter 50` |
| **Validation** | `python -m src.calibration.validation --catchment 204 --calib-year 2010 --years 2018 2019 --mf6 "..."` |
| **Sensitivity (OAT)** | `python -m src.sensitivity.sensitivity_oat --catchment 204 --year 2010 --mf6 "..."` |
| **Sensitivity (Sobol)** | `python -m src.sensitivity.sensitivity_sobol --catchment 204 --year 2010 --mf6 "..."` |
| **Uncertainty (MC)** | `python -m src.sensitivity.uncertainty_mc --catchment 204 --year 2010 --mf6 "..."` |
| **Diagnostics** | `python -m src.diagnostics.diagnostics --data-root "..." --output-dir "..."` |
| **SGD extraction** | `python -m src.diagnostics.sgd_post --cbc "data/output/model_runs/mf6_204/gwf_204.cbc"` |
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
