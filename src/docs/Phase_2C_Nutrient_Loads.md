# Phase 2C: Coastal Nutrient Load Estimation

> **Goal:** Combine SGD fluxes (Phase 2A) with groundwater nutrient concentrations (Phase 2B) to estimate nitrogen and phosphorus loads delivered from groundwater to the coast.

---

## Overview

Nutrient load is the mass of nitrogen (N) or phosphorus (P) transported to the coast per unit time. The fundamental equation is:

$$\text{Load} \; [\text{kg/day}] = \text{SGD flux} \; [\text{m}^3/\text{day}] \times \text{Concentration} \; [\text{kg/m}^3]$$

This is computed cell-by-cell for all coastal (GHB) cells and summed to get the total catchment load. If you have monthly SGD (Phase 2A) and spatially distributed concentrations (Phase 2B), the load can be resolved both in space and time.

---

## What You Need

### From Previous Phases

| Input | Source | Description |
|-------|--------|-------------|
| **SGD fluxes** | Phase 1 or Phase 2A | GHB outflows from MODFLOW budget file (`gwf_<id>.cbc`) |
| **N concentration raster** | Phase 2B | `data/output/cache/<id>/nitrogen_conc_kg_m3.tif` |
| **P concentration raster** | Phase 2B | `data/output/cache/<id>/phosphorus_conc_kg_m3.tif` |
| **GHB cell mask** | Phase 1 model | Boolean mask identifying coastal cells |
| **Model grid info** | Phase 1 | Cell area (from DEM transform) |

### Software

- Python packages: `flopy`, `numpy`, `rasterio`, `pandas`, `matplotlib`
- For uncertainty propagation: `scipy.stats`, `numpy.random`

---

## Step-by-Step Instructions

### Task 2C-1: Compute Nutrient Loads

**Suggested file:** `src/diagnostics/nutrient_loads.py`

#### Step 1: Extract SGD Flux per Coastal Cell

Build on the logic in `sgd_post.py`. For steady-state runs:

```python
import flopy.utils.binaryfile as bf
import numpy as np

cbc_path = f"data/output/model_runs/mf6_{catchment_id}/gwf_{catchment_id}.cbc"
cbc = bf.CellBudgetFile(cbc_path)

# Get GHB flows — the last time step
ghb = cbc.get_data(text="GHB", totim=cbc.get_times()[-1])

# ghb is a list of records; extract the structured array
ghb_rec = ghb[0]

# For steady-state: one record per GHB cell
# ghb_rec has fields: ('node', 'q', ...)
# Negative q = water leaving the aquifer = SGD

# Build a 2D flux array (layer 0 = soil layer)
nrow, ncol = dem.shape  # from cached DEM
sgd_flux = np.zeros((nrow, ncol))
for rec in ghb_rec:
    node = rec['node'] - 1  # 0-indexed
    layer = node // (nrow * ncol)
    remainder = node % (nrow * ncol)
    row = remainder // ncol
    col = remainder % ncol
    q = rec['q']
    if q < 0:  # outflow = SGD
        sgd_flux[row, col] += abs(q)  # m³/day, positive
```

For transient runs (monthly), extract SGD for each stress period:

```python
times = cbc.get_times()
monthly_sgd = {}
for t in times:
    ghb = cbc.get_data(text="GHB", totim=t)
    ghb_rec = ghb[0]
    flux = np.zeros((nrow, ncol))
    for rec in ghb_rec:
        node = rec['node'] - 1
        row = (node % (nrow * ncol)) // ncol
        col = (node % (nrow * ncol)) % ncol
        q = rec['q']
        if q < 0:
            flux[row, col] += abs(q)
    monthly_sgd[t] = flux  # m³/day for that month
```

#### Step 2: Load Concentration Rasters

```python
import rasterio

with rasterio.open(f"data/output/cache/{catchment_id}/nitrogen_conc_kg_m3.tif") as src:
    n_conc = src.read(1)  # kg/m³

with rasterio.open(f"data/output/cache/{catchment_id}/phosphorus_conc_kg_m3.tif") as src:
    p_conc = src.read(1)  # kg/m³
```

#### Step 3: Compute Cell-by-Cell Load

```python
# Steady-state load
n_load_grid = sgd_flux * n_conc   # kg/day per cell
p_load_grid = sgd_flux * p_conc   # kg/day per cell

# Total catchment load
total_n_load = np.nansum(n_load_grid)  # kg/day
total_p_load = np.nansum(p_load_grid)  # kg/day

print(f"N load: {total_n_load:.4f} kg/day  ({total_n_load * 365:.1f} kg/yr)")
print(f"P load: {total_p_load:.4f} kg/day  ({total_p_load * 365:.1f} kg/yr)")
```

#### Step 4: Monthly Load Time Series (if Transient)

```python
import pandas as pd

monthly_loads = []
for month_idx, (t, flux) in enumerate(monthly_sgd.items(), 1):
    n_load = np.nansum(flux * n_conc)
    p_load = np.nansum(flux * p_conc)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month_idx - 1]
    monthly_loads.append({
        'month': month_idx,
        'sgd_m3_day': np.nansum(flux),
        'n_load_kg_day': n_load,
        'p_load_kg_day': p_load,
        'n_load_kg_month': n_load * days_in_month,
        'p_load_kg_month': p_load * days_in_month,
    })

df_loads = pd.DataFrame(monthly_loads)
df_loads['n_load_kg_yr'] = df_loads['n_load_kg_month'].sum()
df_loads['p_load_kg_yr'] = df_loads['p_load_kg_month'].sum()

print(df_loads.to_string(index=False))
df_loads.to_csv(f"data/output/nutrient_loads_{catchment_id}_{year}.csv", index=False)
```

#### Step 5: Visualize Load Distribution

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Spatial map of N load
im0 = axes[0].imshow(np.where(n_load_grid > 0, n_load_grid, np.nan),
                      cmap='YlOrRd', origin='upper')
axes[0].set_title("N load [kg/day per cell]")
plt.colorbar(im0, ax=axes[0])

# Monthly bar chart (if transient)
axes[1].bar(df_loads['month'], df_loads['n_load_kg_month'], color='steelblue', label='N')
axes[1].bar(df_loads['month'], df_loads['p_load_kg_month'], color='coral', label='P',
            bottom=df_loads['n_load_kg_month'])
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Load [kg/month]")
axes[1].legend()
axes[1].set_title("Monthly Nutrient Load")

plt.tight_layout()
plt.savefig(f"data/output/nutrient_load_maps_{catchment_id}_{year}.png", dpi=150)
plt.show()
```

---

### Task 2C-2: Uncertainty Propagation

Both SGD and nutrient concentrations have uncertainties. Propagate them into load estimates.

#### Source 1: SGD Uncertainty

From Phase 1's Monte Carlo analysis (`uncertainty_mc.py`), you have a distribution of SGD values for different parameter sets. Use these as an ensemble of SGD flux realizations.

#### Source 2: Concentration Uncertainty

Nutrient concentrations have measurement variability. For each well:
- Use the standard deviation of repeated measurements as σ
- If only a single measurement, assume a coefficient of variation (e.g., CV = 0.30 for N, 0.50 for P)

#### Monte Carlo Propagation

```python
import numpy.random as rng

N_MC = 1000
rng_gen = rng.default_rng(42)

# From MC ensemble of SGD (Phase 1 uncertainty_mc.py output)
sgd_ensemble = np.load(f"data/output/mc_sgd_ensemble_{catchment_id}.npy")  # shape: (N_mc, nrow, ncol)

# Or if only total SGD values:
sgd_total_samples = np.load(f"data/output/mc_sgd_totals_{catchment_id}.npy")  # shape: (N_mc,)

# Concentration uncertainty: sample from distribution
n_conc_mean = 0.005   # kg/m³ (5 mg/L)
n_conc_std  = 0.0015  # kg/m³

p_conc_mean = 0.0001  # kg/m³ (0.1 mg/L)
p_conc_std  = 0.00005

n_load_samples = []
p_load_samples = []

for i in range(N_MC):
    # Sample SGD
    if sgd_total_samples is not None:
        sgd_i = sgd_total_samples[i % len(sgd_total_samples)]
    else:
        sgd_i = np.nansum(sgd_ensemble[i % len(sgd_ensemble)])

    # Sample concentration (lognormal to prevent negatives)
    n_conc_i = rng_gen.lognormal(
        mean=np.log(n_conc_mean**2 / np.sqrt(n_conc_std**2 + n_conc_mean**2)),
        sigma=np.sqrt(np.log(1 + (n_conc_std/n_conc_mean)**2))
    )
    p_conc_i = rng_gen.lognormal(
        mean=np.log(p_conc_mean**2 / np.sqrt(p_conc_std**2 + p_conc_mean**2)),
        sigma=np.sqrt(np.log(1 + (p_conc_std/p_conc_mean)**2))
    )

    n_load_samples.append(sgd_i * n_conc_i)
    p_load_samples.append(sgd_i * p_conc_i)

n_load_arr = np.array(n_load_samples)
p_load_arr = np.array(p_load_samples)

print(f"N load: {np.median(n_load_arr):.4f} kg/day "
      f"(95% CI: {np.percentile(n_load_arr, 2.5):.4f}–{np.percentile(n_load_arr, 97.5):.4f})")
print(f"P load: {np.median(p_load_arr):.6f} kg/day "
      f"(95% CI: {np.percentile(p_load_arr, 2.5):.6f}–{np.percentile(p_load_arr, 97.5):.6f})")
```

#### Simplified Analytical Propagation

If a full MC is not needed, use error propagation for products:

$$\frac{\sigma_{\text{Load}}}{\text{Load}} = \sqrt{\left(\frac{\sigma_{\text{SGD}}}{\text{SGD}}\right)^2 + \left(\frac{\sigma_C}{C}\right)^2}$$

```python
sgd_mean, sgd_std = 500.0, 75.0   # m³/day from MC results
n_mean, n_std = 0.005, 0.0015     # kg/m³

n_load_mean = sgd_mean * n_mean
n_load_cv = np.sqrt((sgd_std/sgd_mean)**2 + (n_std/n_mean)**2)
n_load_std = n_load_mean * n_load_cv

print(f"N load: {n_load_mean:.4f} ± {n_load_std:.4f} kg/day")
```

---

## Expected Output Files

```
data/output/
├── nutrient_loads_<catchment_id>_<year>.csv        ← monthly/annual N and P loads
├── nutrient_load_maps_<catchment_id>_<year>.png    ← spatial + temporal plots
├── mc_nutrient_loads_<catchment_id>.npy            ← MC ensemble of load values
└── nutrient_load_summary.csv                       ← multi-catchment comparison table
```

---

## Sanity Check — Expected Magnitudes

For a small Swedish coastal catchment (~50–200 km²):

| Metric | N | P |
|--------|---|---|
| SGD flux | 100–2000 m³/day | — |
| Concentration | 1–15 mg/L | 0.01–0.3 mg/L |
| Load | 0.1–30 kg/day | 0.001–0.5 kg/day |
| Annual load | 30–10,000 kg/yr | 0.5–200 kg/yr |

If your numbers are orders of magnitude outside these ranges, double-check units and concentration inputs.

---

## Verification Checklist

- [ ] SGD fluxes extracted correctly (only negative q from GHB = outflow)
- [ ] Concentration rasters aligned with model grid (same shape, CRS, transform)
- [ ] Unit consistency: SGD in m³/day, concentration in kg/m³, load in kg/day
- [ ] Total load = sum of cell-by-cell loads
- [ ] Monthly loads × days/month = total monthly mass
- [ ] Annual load = sum of 12 monthly loads
- [ ] Uncertainty bounds reported (at minimum, 95% CI)
- [ ] Loads are physically reasonable (see sanity check above)
- [ ] Results saved as CSV and plotted

---

## Key References

- UNESCO-IHP (2004): Submarine groundwater discharge — nutrient loading review
- Burnett et al. (2006): Quantifying submarine groundwater discharge in the coastal zone via multiple methods. *Science of the Total Environment*, 367, 498–543.
- Destouni & Prieto (2003): On the possibility for generic modeling of submarine groundwater discharge. *Biogeochemistry*, 66, 171–186.
