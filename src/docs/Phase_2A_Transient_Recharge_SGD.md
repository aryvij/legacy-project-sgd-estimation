# Phase 2A: Time-Varying Recharge & Transient SGD

> **Goal:** Convert the current pseudo-transient (annual) model into a fully transient monthly simulation so that seasonal SGD can be estimated.

---

## Overview

The existing model uses a single annual recharge raster per year, applied across 2 stress periods (30-day ramp at 40% + 335-day run at 100%). Phase 2A replaces this with 12 monthly stress periods, each receiving its own scaled recharge array.

---

## What You Need

### Data

| Item | Description | Where to Get It | Format |
|------|-------------|-----------------|--------|
| **Annual recharge rasters** | Already available from Phase 1 | `data/output/recharge_yearly/recharge_egdi_gldas_<YEAR>.tif` | GeoTIFF (mm/yr) |
| **Monthly precipitation** | Monthly P totals for Sweden | [SMHI Open Data](https://opendata-download-metobs.smhi.se/) — parameter 5 (monthly precip) | CSV per station, or gridded NetCDF |
| **Monthly ET** | Monthly actual evapotranspiration | SMHI or ERA5/GLDAS reanalysis | NetCDF or GeoTIFF |
| **Monthly sea level** (optional) | If GHB stage should vary monthly | [SMHI Oceanographic Obs](https://www.smhi.se/data/oceanografi/ladda-ner-oceanografiska-observationer) | CSV |

### Software

- Python packages: `flopy`, `rasterio`, `numpy`, `pandas`, `xarray` (if using NetCDF)
- MODFLOW 6 executable (same as Phase 1)

---

## Step-by-Step Instructions

### Task 2A-1: Compute Monthly Recharge Factors

**Objective:** Derive 12 monthly scaling factors so that `recharge_month_i = annual_recharge × factor_i` and `sum(factors) = 1.0`.

**Approach A — From SMHI precipitation data:**

1. Download monthly precipitation data for stations within or near the study catchments from SMHI.
2. For each month, compute the fraction of annual precipitation that falls in that month:
   ```
   factor_month = P_month / P_annual
   ```
3. Average across stations and years (or use a representative year if spatial variability is low).
4. Subtract monthly ET if available:
   ```
   R_month = P_month - ET_month
   factor_month = max(R_month, 0) / sum(max(R_month, 0) for all months)
   ```

**Approach B — From literature:**

Use published seasonal recharge distributions for Scandinavian/Nordic climates. Typical pattern:

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Factor | 0.05 | 0.04 | 0.12 | 0.18 | 0.10 | 0.03 | 0.02 | 0.02 | 0.05 | 0.10 | 0.15 | 0.14 |

> Spring snowmelt (Mar–Apr) and autumn rains (Oct–Nov) dominate Swedish recharge.

**Output:** A dict or CSV of 12 monthly factors that sum to 1.0.

**Suggested file:** `src/core/monthly_recharge.py`

```python
# Example structure
MONTHLY_FACTORS = {
    1: 0.05, 2: 0.04, 3: 0.12, 4: 0.18, 5: 0.10, 6: 0.03,
    7: 0.02, 8: 0.02, 9: 0.05, 10: 0.10, 11: 0.15, 12: 0.14
}

def compute_monthly_recharge(annual_rech_array, month):
    """Scale annual recharge (m/day) to a monthly rate."""
    factor = MONTHLY_FACTORS[month]
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    # annual_rech is in m/day (average over 365 days)
    # monthly rate = annual_total * factor / days_in_month
    monthly_rate = annual_rech_array * 365.0 * factor / days_in_month
    return monthly_rate
```

---

### Task 2A-2: Modify modflow_setup.py for Monthly Stress Periods

**Objective:** Change TDIS from 2 stress periods to 12, and assign per-month recharge.

**What to change in `src/core/modflow_setup.py`:**

1. **TDIS definition** (currently around line 486):
   ```python
   # CURRENT (pseudo-transient, 2 stress periods):
   periods = [
       (30.0,  30, 1.0),
       (335.0, 50, 1.0),
   ]
   flopy.mf6.ModflowTdis(sim, nper=2, time_units='days', perioddata=periods)
   ```
   
   **Change to:**
   ```python
   # TRANSIENT (12 monthly stress periods):
   days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
   periods = [(d, max(d // 3, 1), 1.0) for d in days_per_month]
   flopy.mf6.ModflowTdis(sim, nper=12, time_units='days', perioddata=periods)
   ```

2. **Recharge package** (currently around line 585):
   ```python
   # CURRENT:
   rch_spd = {
       0: 0.4 * rech_mf,
       1: 1.0 * rech_mf,
   }
   flopy.mf6.ModflowGwfrcha(gwf, recharge=rch_spd)
   ```
   
   **Change to:**
   ```python
   # TRANSIENT (per-month recharge):
   from core.monthly_recharge import compute_monthly_recharge
   rch_spd = {}
   for sp in range(12):
       month = sp + 1  # 1-based month
       monthly_rech = compute_monthly_recharge(rech_scaled, month)
       rch_spd[sp] = np.nan_to_num(monthly_rech, nan=0.0)
   flopy.mf6.ModflowGwfrcha(gwf, recharge=rch_spd)
   ```

3. **Storage package** — already configured as transient (good). Mark all 12 periods as transient:
   ```python
   flopy.mf6.ModflowGwfsto(
       gwf,
       iconvert=[1, 0],
       sy=[sy1, np.zeros_like(sy1)],
       ss=[ss1*np.ones_like(sy1), ss2*np.ones_like(sy1)],
       steady_state={k: False for k in range(12)},
       transient={k: True for k in range(12)},
   )
   ```

4. **Boundary conditions (GHB, RIV, DRN):** These currently use `{0: spd_list}` which means the same data repeats for all stress periods. This is fine for a first pass. If you have monthly sea-level data, update GHB stage per stress period.

5. **Output control** — save heads for ALL time steps (not just LAST):
   ```python
   flopy.mf6.ModflowGwfoc(gwf,
       head_filerecord=f"gwf_{catchment_id}.hds",
       budget_filerecord=f"gwf_{catchment_id}.cbc",
       saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')]
   )
   ```

**Tip:** Add a new parameter to `setup_and_run_modflow()`:
```python
def setup_and_run_modflow(
    ...,
    transient_monthly: bool = False,   # NEW FLAG
    monthly_factors: dict | None = None,
):
```
This lets you keep backward compatibility — `transient_monthly=False` uses the old 2-SP setup.

---

### Task 2A-3: Extract Seasonal Groundwater Heads

**Objective:** Read monthly head fields from the HDS file and compare against observed well data.

**How to read monthly heads:**
```python
import flopy
hf = flopy.utils.HeadFile("gwf_204.hds")
times = hf.get_times()  # list of times at end of each stress period

for i, t in enumerate(times):
    month = i + 1
    heads_3d = hf.get_data(totim=t)  # shape (nlay, nrow, ncol)
    # heads_3d[0] = Layer 1 (soil), heads_3d[1] = Layer 2 (rock)
    # Save or plot as needed
```

**Validation approach:**
- Filter well observations (`brunnar.gpkg`) by month using the `nivadatum` field
- Compare simulated head at each well location/month vs. observed
- Compute seasonal RMSE and plot monthly head time series at selected well locations

---

### Task 2A-4: Extract Monthly SGD

**Objective:** Extract SGD (GHB outflows) for each month from the budget file.

**Modify `src/diagnostics/sgd_post.py`** to loop over all time steps:

```python
def extract_monthly_sgd(base_ws, catchment):
    """Extract SGD for each stress period (month)."""
    cbc_path = os.path.join(base_ws, f"gwf_{catchment}.cbc")
    cbc = flopy.utils.CellBudgetFile(cbc_path, precision='double')
    times = cbc.get_times()
    
    results = []
    for i, t in enumerate(times):
        ghb = cbc.get_data(text="GHB", totim=t)
        if ghb:
            q = ghb[-1]['q'].astype(float)
            sgd = float(np.abs(np.nansum(q[q < 0])))
        else:
            sgd = 0.0
        results.append({
            "month": i + 1,
            "totim_days": t,
            "sgd_m3_per_day": sgd,
        })
    return results
```

**Expected output:** A table/plot showing SGD by month. Expect peaks during spring (high recharge from snowmelt) and potentially autumn (rain).

---

## Verification Checklist

- [ ] Monthly factors sum to 1.0
- [ ] TDIS has 12 stress periods with correct day counts
- [ ] Recharge array varies by month (check min/max per SP)
- [ ] Model converges for all 12 stress periods (check listing file for convergence)
- [ ] Budget file has 12 time entries
- [ ] Monthly SGD values are physically reasonable (same order of magnitude as annual average)
- [ ] Annual sum of monthly SGD ≈ pseudo-transient SGD from Phase 1

---

## Key References

- FloPy TDIS: https://flopy.readthedocs.io/en/latest/source/flopy.mf6.modflow.mfgwftdis.html
- FloPy budget reading: https://flopy.readthedocs.io/en/latest/source/flopy.utils.binaryfile.html
- SMHI precipitation data: https://opendata-download-metobs.smhi.se/
- ERA5 reanalysis (alternative for monthly P/ET): https://cds.climate.copernicus.eu/
