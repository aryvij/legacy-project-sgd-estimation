# Phase 2D: Groundwater Transport Modelling (MODFLOW 6 GWT)

> **Goal:** Set up a MODFLOW 6 Groundwater Transport (GWT) model coupled with the existing Groundwater Flow (GWF) model to simulate advective–dispersive transport of nutrients (N, P) through the aquifer to the coast.

> **Note:** This phase is optional. If data and time constraints allow only the simpler load calculation (Phase 2C), that approach is still scientifically valid. GWT adds spatial and temporal realism at the cost of additional parameterization and calibration.

---

## Overview

MODFLOW 6 supports a coupled GWF–GWT simulation where:
- **GWF** solves the flow equation (already done in Phase 1)
- **GWT** solves the advection–dispersion equation (ADE) for a dissolved species

The GWT model uses the flow solution from GWF (heads, cell-by-cell flows) to transport a solute through the same grid. You can model N and P as separate GWT models coupled to the same GWF model.

Key packages:
- **MST** — Mobile Storage and Transfer (porosity, sorption, decay)
- **ADV** — Advection (numerical scheme)
- **DSP** — Dispersion (longitudinal, transverse, molecular diffusion)
- **SSM** — Source/Sink Mixing (assigns concentrations to flow sources)
- **CNC** — Constant Concentration (Dirichlet boundary)
- **SRC** — Mass Source (adds mass at cells)
- **OC** — Output Control for concentrations

---

## What You Need

### Data

| Input | Description | Typical Values for Sweden | Source |
|-------|-------------|---------------------------|--------|
| **Porosity** (soil layer) | Effective porosity of soil | 0.15–0.40 | Literature / SGU soil type |
| **Porosity** (rock layer) | Effective porosity of fractured bedrock | 0.005–0.05 | Literature |
| **Longitudinal dispersivity** | Along-flow dispersion | 10–100 m (scale-dependent) | Gelhar et al., 1992 |
| **Transverse dispersivity** | Cross-flow dispersion | 0.1× longitudinal | Literature |
| **Molecular diffusion** | Diffusion coefficient in water | ~1×10⁻⁹ m²/s (~8.6×10⁻⁵ m²/day) | Standard |
| **Recharge concentration** | N and P in infiltrating water | Varies (from land-use / atmospheric dep.) | SGU / SMHI / literature |
| **Initial concentration** | Starting N and P in groundwater | From Phase 2B well data | Processed wells |
| **Observed well concentrations** | For calibration/validation | From Phase 2B | SGU chemistry data |
| **Decay rate** (optional) | First-order denitrification rate | 0.001–0.01 day⁻¹ for N | Literature |

### Software

- **FloPy** (≥ 3.4.0) — has full GWT support
- **MODFLOW 6** (≥ 6.4.0) — ensure your binary includes GWT
- Python: `flopy`, `numpy`, `rasterio`, `matplotlib`, `scipy`

---

## Step-by-Step Instructions

### Task 2D-1: Set Up the GWT Model

**Suggested file:** `src/core/transport_setup.py`

#### Step 1: Understand the Simulation Structure

In MODFLOW 6, a coupled GWF–GWT simulation looks like:

```
mfsim.nam
├── gwf_204.nam        ← existing GWF model
├── gwt_n_204.nam      ← new GWT model for nitrogen
└── gwf_204-gwt_n_204  ← GWF–GWT exchange (links flow → transport)
```

The GWT model must use the **same grid** (DIS) and **same time discretization** (TDIS) as the GWF model.

#### Step 2: Create the GWT Model in FloPy

```python
import flopy
import numpy as np

# Load the existing simulation
sim_ws = f"data/output/model_runs/mf6_{catchment_id}"
sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
gwf = sim.get_model(f"gwf_{catchment_id}")

# Get grid dimensions from the GWF model
nlay = gwf.dis.nlay.data
nrow = gwf.dis.nrow.data
ncol = gwf.dis.ncol.data

# Create a GWT model for nitrogen
gwt_name = f"gwt_n_{catchment_id}"
gwt = flopy.mf6.MFModel(
    sim,
    model_type="gwt6",
    modelname=gwt_name,
)

# DIS — same discretization as GWF
flopy.mf6.ModflowGwtdis(
    gwt,
    nlay=nlay, nrow=nrow, ncol=ncol,
    delr=gwf.dis.delr.data,
    delc=gwf.dis.delc.data,
    top=gwf.dis.top.data,
    botm=gwf.dis.botm.data,
    idomain=gwf.dis.idomain.data,
)
```

#### Step 3: Initial Conditions (IC)

Set initial concentrations from Phase 2B:

```python
import rasterio

# Load concentration raster (Phase 2B output)
with rasterio.open(f"data/output/cache/{catchment_id}/nitrogen_conc_kg_m3.tif") as src:
    n_conc_2d = src.read(1)  # kg/m³

# Convert to mg/L for GWT (or keep consistent — pick one unit system)
# 1 kg/m³ = 1000 mg/L
n_conc_mg_L = n_conc_2d * 1000.0

# Set for both layers (or differentiate if you have depth data)
strt_conc = np.zeros((nlay, nrow, ncol))
strt_conc[0, :, :] = n_conc_mg_L              # soil layer
strt_conc[1, :, :] = n_conc_mg_L * 0.5        # rock layer (typically lower)

flopy.mf6.ModflowGwtic(gwt, strt=strt_conc)
```

#### Step 4: Mobile Storage and Transfer (MST)

```python
# Porosity arrays
porosity = np.zeros((nlay, nrow, ncol))
porosity[0, :, :] = 0.25   # soil layer
porosity[1, :, :] = 0.01   # fractured rock

# Optional: first-order decay (denitrification for N)
decay_rate = np.zeros((nlay, nrow, ncol))
decay_rate[0, :, :] = 0.005  # day⁻¹ (soil — active denitrification)
decay_rate[1, :, :] = 0.001  # day⁻¹ (rock — slower)

flopy.mf6.ModflowGwtmst(
    gwt,
    porosity=porosity,
    first_order_decay=True,
    decay=decay_rate,
    # Optional sorption:
    # sorption="linear",
    # bulk_density=1800,  # kg/m³
    # distcoef=0.0001,    # L/kg distribution coefficient
)
```

#### Step 5: Advection (ADV)

```python
# TVD (Total Variation Diminishing) is recommended for accuracy
flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")
```

#### Step 6: Dispersion (DSP)

```python
flopy.mf6.ModflowGwtdsp(
    gwt,
    alh=50.0,         # longitudinal dispersivity [m] (horizontal)
    ath1=5.0,          # transverse dispersivity [m] (horizontal)
    atv=1.0,           # transverse dispersivity [m] (vertical)
    diffc=8.64e-5,     # molecular diffusion [m²/day]
)
```

**Note on dispersivity:** This is highly scale-dependent. For a regional model with cells > 100 m, longitudinal dispersivity of 50–100 m is typical. For finer grids, use smaller values.

#### Step 7: Source/Sink Mixing (SSM)

SSM automatically assigns concentrations to flow boundary sources (RCH, GHB, RIV, DRN). You specify what concentration each source carries:

```python
# Build SSM source list
# Each entry: (package_name, "AUX", aux_variable_name) or explicit stress period data

# For recharge: specify N concentration in infiltrating water
rch_conc = 0.005  # kg/m³ = 5 mg/L (agricultural leachate)

# For GHB (seawater): typically low N
ghb_conc = 0.001  # kg/m³ = 1 mg/L (seawater background)

# For RIV (river water): intermediate
riv_conc = 0.003  # kg/m³ = 3 mg/L

# SSM sources — reference the auxiliary CONCENTRATION variable
# This requires adding "CONCENTRATION" as an auxiliary variable
# to the GWF boundary packages (RCH, GHB, RIV)
# This is the recommended approach in MODFLOW 6

sources = [
    ("RCH-1", "AUX", "CONCENTRATION"),
    ("GHB-1", "AUX", "CONCENTRATION"),
    ("RIV-1", "AUX", "CONCENTRATION"),
]
flopy.mf6.ModflowGwtssm(gwt, sources=sources)
```

**Important:** To use SSM with auxiliary concentrations, you must modify the GWF model's RCH, GHB, and RIV packages to include a `CONCENTRATION` auxiliary variable. This means modifying `modflow_setup.py`:

```python
# Example: adding CONCENTRATION aux to RCH in modflow_setup.py
flopy.mf6.ModflowGwfrcha(
    gwf,
    recharge=rch_array,
    auxiliary=["CONCENTRATION"],
    aux={0: [rch_conc]},  # concentration for stress period 0
)
```

#### Step 8: Output Control (OC)

```python
flopy.mf6.ModflowGwtoc(
    gwt,
    concentration_filerecord=f"{gwt_name}.ucn",
    concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
    saverecord=[("CONCENTRATION", "LAST")],
    printrecord=[("CONCENTRATION", "LAST")],
)
```

#### Step 9: GWF–GWT Exchange

Link the flow model to the transport model:

```python
flopy.mf6.ModflowGwfgwt(
    sim,
    exgtype="GWF6-GWT6",
    exgmnamea=f"gwf_{catchment_id}",
    exgmnameb=gwt_name,
)
```

#### Step 10: Write and Run

```python
sim.write_simulation()
sim.run_simulation()
```

---

### Task 2D-2: Transport Calibration & Validation

#### Calibration Parameters

| Parameter | Range | Sensitivity |
|-----------|-------|-------------|
| Porosity (soil) | 0.10–0.40 | High |
| Porosity (rock) | 0.005–0.05 | Medium |
| Longitudinal dispersivity | 10–200 m | High |
| Transverse dispersivity | 1–20 m | Low–Medium |
| Decay rate (N only) | 0–0.02 day⁻¹ | High for N |
| Recharge concentration | 0.001–0.020 kg/m³ | High |

#### Calibration Approach

1. **Objective function:** RMSE of simulated vs. observed concentrations at well locations.

```python
def transport_rmse(params, sim, gwt_name, obs_wells):
    porosity_soil, porosity_rock, disp_l, decay_n = params

    gwt = sim.get_model(gwt_name)

    # Update parameters
    gwt.mst.porosity.set_data([porosity_soil, porosity_rock])
    gwt.mst.decay.set_data([decay_n, decay_n * 0.2])
    gwt.dsp.alh.set_data(disp_l)
    gwt.dsp.ath1.set_data(disp_l * 0.1)

    sim.write_simulation()
    sim.run_simulation()

    # Read simulated concentrations
    ucn = flopy.utils.HeadFile(f"{sim.sim_path}/{gwt_name}.ucn", text="CONCENTRATION")
    conc = ucn.get_data(totim=ucn.get_times()[-1])

    # Compare to observed
    residuals = []
    for _, well in obs_wells.iterrows():
        row, col = well['row'], well['col']
        layer = 0  # assume soil layer
        sim_c = conc[layer, row, col]
        obs_c = well['conc_mg_L']
        residuals.append(sim_c - obs_c)

    return np.sqrt(np.mean(np.array(residuals)**2))
```

2. **Optimizer:** Use the same Nelder-Mead approach as the head calibration:

```python
from scipy.optimize import minimize

x0 = [0.25, 0.01, 50.0, 0.005]
bounds = [(0.10, 0.40), (0.005, 0.05), (10.0, 200.0), (0.0, 0.02)]

result = minimize(transport_rmse, x0, args=(sim, gwt_name, obs_wells),
                  method='Nelder-Mead',
                  options={'maxiter': 30, 'xatol': 0.01, 'fatol': 0.1})
```

3. **Validation:** Run with calibrated transport parameters on unseen years and compare simulated concentrations to independent observations.

#### Reading Transport Results

```python
# Read concentration output
ucn_path = f"data/output/model_runs/mf6_{catchment_id}/{gwt_name}.ucn"
ucn = flopy.utils.HeadFile(ucn_path, text="CONCENTRATION")

# Final concentrations
conc_final = ucn.get_data(totim=ucn.get_times()[-1])
print(f"Concentration range: {conc_final[conc_final > 0].min():.4f} – "
      f"{conc_final.max():.4f} mg/L")

# Extract coastal cell concentrations (GHB cells)
coastal_conc = conc_final[0][ghb_mask]  # soil layer, coastal cells
print(f"Mean coastal N concentration: {np.mean(coastal_conc):.4f} mg/L")
```

#### Nutrient Load from GWT

With GWT, the nutrient load at the coast comes directly from the transport solution — no separate multiplication needed:

```python
# Mass flux at GHB cells = flow × simulated concentration
# This is already handled by MODFLOW 6 internally
# Extract from the GWT budget file:
gwt_cbc = bf.CellBudgetFile(f"{sim.sim_path}/{gwt_name}.cbc")
ssm_ghb = gwt_cbc.get_data(text="GHB", totim=gwt_cbc.get_times()[-1])

# Mass outflow (negative = leaving to coast)
total_mass_flux = sum(rec['q'] for rec in ssm_ghb[0] if rec['q'] < 0)
print(f"N mass flux to coast: {abs(total_mass_flux):.4f} mass_unit/day")
```

---

## Implementation Order

1. **Start with steady-state GWT** (one stress period for simplicity)
2. Get the basic transport model running with uniform parameters
3. Add decay for nitrogen only (phosphorus is typically conservative or sorption-dominated)
4. Calibrate against observed well concentrations
5. Extend to transient (monthly) if flow model is also transient
6. Run for phosphorus as a separate GWT model (different parameters)

---

## Modifying the Existing Code

The key file to modify is `modflow_setup.py`. Changes needed:

1. **Add auxiliary `CONCENTRATION` to boundary packages** (GHB, RIV, RCH, DRN)
2. **Add GWT model creation** after the GWF model is built
3. **Add GWF–GWT exchange** to the simulation
4. **Pass transport parameters** through the function signature

Consider creating a separate function `setup_transport_model()` in `transport_setup.py` that takes the GWF model/simulation and adds GWT on top, rather than modifying `modflow_setup.py` extensively.

---

## Common Pitfalls

| Issue | Solution |
|-------|----------|
| Mass balance errors > 1% | Reduce time step size, check DSP/ADV settings |
| Negative concentrations | Use TVD advection scheme, not upstream |
| Very slow convergence | Increase OUTER_MAXIMUM in IMS for GWT |
| Concentrations don't change | Check porosity values — too-high porosity delays transport |
| GWT crashes with dry cells | Ensure GWF uses rewetting or Newton solver |
| Units mismatch | Pick one system (mg/L or kg/m³) and be consistent everywhere |

---

## Expected Output Files

```
data/output/model_runs/mf6_<catchment_id>/
├── gwt_n_<id>.ucn          ← concentration binary output (N)
├── gwt_n_<id>.cbc          ← mass budget binary output (N)
├── gwt_p_<id>.ucn          ← concentration binary output (P)
└── gwt_p_<id>.cbc          ← mass budget binary output (P)

data/output/
├── transport_calibration_<id>.csv     ← calibration results
├── coastal_nutrient_flux_gwt_<id>.csv ← mass flux at coast from GWT
└── transport_diagnostics_<id>.png     ← concentration maps + obs comparison
```

---

## Verification Checklist

- [ ] GWT model uses same grid (DIS) and time discretization (TDIS) as GWF
- [ ] GWF–GWT exchange registered in simulation
- [ ] Porosity values are physically reasonable (soil 0.15–0.40, rock 0.005–0.05)
- [ ] Dispersivity is scale-appropriate (≈ 1/10 to 1/3 of grid spacing)
- [ ] Initial concentrations match Phase 2B processed data
- [ ] Source concentrations assigned to all boundary packages (SSM)
- [ ] Mass balance error < 1% (check listing file)
- [ ] Simulated concentrations compared to observed well data
- [ ] Coastal mass flux extracted and reported
- [ ] N and P run as separate GWT models
- [ ] Decay only applied to N (not P, unless justified)

---

## Key References

- MODFLOW 6 GWT documentation: https://modflow6.readthedocs.io/en/latest/
- FloPy GWT examples: https://flopy.readthedocs.io/en/latest/
- Langevin et al. (2022): MODFLOW 6 Modular Hydrologic Model version 6.4.0. USGS.
- Gelhar et al. (1992): A critical review of data on field-scale dispersion in aquifers. *Water Resources Research*, 28(7), 1955–1974.
- Zheng & Wang (1999): MT3DMS: A modular three-dimensional multispecies transport model. US Army Corps of Engineers.
