Calibration of MODFLOW-6 Model for SGD Catchment 204 (Year 2018)
1. Objective

The aim of the calibration was to adjust model parameters to reproduce observed groundwater heads in catchment 204 (year 2018). Calibration was performed against interpolated head surfaces from well observations. The key objective function was the root-mean-square error (RMSE) between simulated and observed heads.

2. Data and Inputs

Topography: Digital Elevation Model (DEM, 10 m resolution).

Soil data: Soil depth raster and soil permeability classes.

Rock hydraulic conductivity: Raster of hydraulic conductivity (m/s).

Recharge: Year-specific raster derived from EGDI and GLDAS (m/day).

Hydrography: Coastline and river shapefiles.

Wells: National groundwater well database (brunnar.gpkg).

3. Calibration Parameters

Multipliers were defined for five uncertain parameters:

Soil-K multiplier

Rock-K multiplier

River conductance multiplier (RIV)

General head boundary conductance multiplier (GHB)

Recharge multiplier (RCH)

These multipliers allow adjustment of parameter fields while preserving their spatial patterns.

4. Early Attempts

All multipliers were set free for optimization.

Optimization consistently plateaued at RMSE ≈ 9 m.

No significant convergence was observed: the Powell optimizer flatlined after a few iterations.

Problem identified: strong equifinality — different combinations of multipliers yielded the same RMSE, meaning the parameters were not identifiable given only head data.

5. Issues Identified

Head hot-spots: Simulated heads spiked in areas with high recharge but zero/near-zero rock hydraulic conductivity. These unrealistic water “bathtubs” strongly biased calibration.

Equifinality: With five free multipliers, optimization space was flat and non-unique.

Residual distribution: Errors concentrated in a recharge/K hotspot zone, masking meaningful calibration elsewhere.

6. Fixes Applied
A. Rock-K Flooring

Action: Preprocessed hydraulic_conductivity.tif into hydraulic_conductivity_floored.tif.

All values ≤ 1×10⁻⁶ m/s replaced with that floor.

Rationale: Prevents recharge-rich cells from being completely impermeable, enabling drainage and reducing artificial head build-up.

B. Hotspot Mask

Action: Created binary GeoTIFF mask for the hotspot zone (1 = hotspot, 0 = elsewhere).

Use: Optionally excluded hotspot cells from RMSE calculation while still saving residuals for transparency.

Rationale: Prevents known problematic areas from dominating calibration but retains them for diagnostic reporting.

C. Restricting Calibration Parameters

Action: Fixed soil-K, rock-K, RIV, and GHB multipliers at 1.0.

Allowed only the recharge multiplier to vary during calibration.

Rationale: Recharge is the dominant identifiable parameter at catchment scale; other multipliers caused equifinality and non-convergence.

7. Current Calibration Protocol

Calibration is performed with the following command: