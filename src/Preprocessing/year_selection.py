#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Catchment Recharge & Wells Summary (clean)

Part A — Recharge (CSV, no catchment filter):
- Wettest year (highest mean Recharge_mm_year)
- Driest year (lowest mean Recharge_mm_year)
- 5 years whose mean recharge are closest to the median of yearly means (sorted by distance)

Part B — Wells (GPKG + catchment shapefile):
- Asks for ID_BSDB
- For that catchment, computes:
  • Year(s) with most wells
  • Year(s) with least wells
  • Counts for years 2000–2025 (zero-filled)

Paths:
  CSV_RECHARGE  = C:/Users/aryapv/OneDrive - KTH/Modelling_SGD_Arya/SGD_model/data/input/recharge_data_selection_for_calibration.csv
  SHP_CATCHMENTS= C:/Users/aryapv/OneDrive - KTH/Modelling_SGD_Arya/SGD_model/data/input/shapefiles/catchment/bsdbs.shp
  GPKG_WELLS    = C:/Users/aryapv/OneDrive - KTH/Modelling_SGD_Arya/SGD_model/data/input/well_data/brunnar.gpkg

Notes:
- Wells layer is explicitly set to "brunnar" to avoid multi-layer warnings.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# === Fixed paths ===
CSV_RECHARGE = r"C:/Users/aryapv/OneDrive - KTH/Modelling_SGD_Arya/SGD_model/data/input/recharge_data_selection_for_calibration.csv"
SHP_CATCHMENTS = r"C:/Users/aryapv/OneDrive - KTH/Modelling_SGD_Arya/SGD_model/data/input/shapefiles/catchment/bsdbs.shp"
GPKG_WELLS   = r"C:/Users/aryapv/OneDrive - KTH/Modelling_SGD_Arya/SGD_model/data/input/well_data/brunnar.gpkg"
WELLS_LAYER  = "brunnar"

OUTPUT_DIR = "output"

# ---------- Helpers ----------

def read_csv_any_delim(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def pick_first(cols, options):
    for c in options:
        if c in cols:
            return c
    return None

# ---------- Recharge (overall) ----------


def summarize_recharge_overall(csv_path):
    df = read_csv_any_delim(csv_path)

    # Normalize columns
    lower = {c: c.lower().strip() for c in df.columns}
    inv   = {v: k for k, v in lower.items()}

    year_key = pick_first(lower.values(), ["year", "yr"])
    rech_key = pick_first(lower.values(), ["recharge_mm_year", "recharge", "recharge_mm"])

    missing = []
    if year_key is None: missing.append("year")
    if rech_key is None: missing.append("Recharge_mm_year")
    if missing:
        raise ValueError(f"Recharge CSV missing columns: {', '.join(missing)}. Found: {list(df.columns)}")

    year_col = inv[year_key]
    rech_col = inv[rech_key]

    use = df[[year_col, rech_col]].dropna()
    use = use.rename(columns={year_col: "year", rech_col: "Recharge_mm_year"})
    use["year"] = use["year"].astype(int)
    use["Recharge_mm_year"] = use["Recharge_mm_year"].astype(float)

    per_year = use.groupby("year")["Recharge_mm_year"].agg(["mean", "median", "count"]).reset_index()
    per_year = per_year.rename(columns={"mean": "mean_recharge", "median": "median_recharge", "count": "n_points"})

    # Wettest/driest
    max_mean = per_year["mean_recharge"].max()
    min_mean = per_year["mean_recharge"].min()
    wettest_years = per_year.loc[per_year["mean_recharge"] == max_mean, "year"].astype(int).tolist()
    driest_years  = per_year.loc[per_year["mean_recharge"] == min_mean, "year"].astype(int).tolist()

    # Median-based (5 closest)
    overall_median = float(np.median(per_year["mean_recharge"]))
    per_year["abs_diff_med"] = (per_year["mean_recharge"] - overall_median).abs()
    closest_to_median_years = (
        per_year.sort_values(["abs_diff_med", "year"])
        .head(5)["year"].astype(int).tolist()
    )

    # Closest to wettest (exclude the wettest year(s) themselves)
    per_year["abs_diff_wet"] = (per_year["mean_recharge"] - max_mean).abs()
    closest_to_wettest_years = (
        per_year.loc[~per_year["year"].isin(wettest_years)]
                .sort_values(["abs_diff_wet", "year"])
                .head(5)["year"].astype(int).tolist()
    )

    # Closest to driest (exclude the driest year(s) themselves)
    per_year["abs_diff_dry"] = (per_year["mean_recharge"] - min_mean).abs()
    closest_to_driest_years = (
        per_year.loc[~per_year["year"].isin(driest_years)]
                .sort_values(["abs_diff_dry", "year"])
                .head(5)["year"].astype(int).tolist()
    )

    per_year = (per_year
                .drop(columns=["abs_diff_med", "abs_diff_wet", "abs_diff_dry"])
                .sort_values("year")
                .reset_index(drop=True))

    summary = {
        "wettest_years": wettest_years,
        "wettest_mean_recharge": float(max_mean),
        "driest_years": driest_years,
        "driest_mean_recharge": float(min_mean),
        "overall_median_of_yearly_means": overall_median,
        "closest_to_median_years": closest_to_median_years,
        "closest_to_wettest_years": closest_to_wettest_years,
        "closest_to_driest_years": closest_to_driest_years,
    }
    return per_year, summary

    # ---------- Catchments & wells ----------

def prompt_catchment_id(gdf_catch):
    print("\nEnter the ID_BSDB you want to analyze (examples):")
    for cid in pd.Series(gdf_catch["ID_BSDB"].astype(str).unique()).head(20):
        print("  -", cid)
    return input("> ").strip()

def load_catchment(shp_path, catchment_id):
    cats = gpd.read_file(shp_path)
    if "ID_BSDB" not in cats.columns:
        raise ValueError("Catchment shapefile must contain 'ID_BSDB'.")
    cats["ID_BSDB"] = cats["ID_BSDB"].astype(str)
    sel = cats[cats["ID_BSDB"] == str(catchment_id)]
    if sel.empty:
        raise ValueError(f"Catchment ID {catchment_id} not found.")
    if cats.crs is None:
        cats = cats.set_crs("EPSG:4326", allow_override=True)
        sel = cats[cats["ID_BSDB"] == str(catchment_id)]
    return sel.dissolve(by="ID_BSDB", as_index=False)

def load_wells(gpkg_path, layer=WELLS_LAYER):
    # Read the specified layer to avoid multi-layer warnings
    gdf = gpd.read_file(gpkg_path, layer=layer)
    if gdf.geometry.isna().all():
        # Build from e/n
        cols = {c.lower(): c for c in gdf.columns}
        e_col = cols.get("e")
        n_col = cols.get("n")
        if e_col is None or n_col is None:
            raise ValueError("Wells file has no geometry and is missing 'e'/'n' columns.")
        gdf = gdf.copy()
        gdf["e"] = gdf[e_col].astype(float)
        gdf["n"] = gdf[n_col].astype(float)
        gdf = gpd.GeoDataFrame(gdf, geometry=[Point(xy) for xy in zip(gdf["e"], gdf["n"])], crs="EPSG:3006")
    elif gdf.crs is None:
        # If geometry exists but CRS is missing, assume EPSG:3006 (SWEREF99 TM)
        gdf = gdf.set_crs("EPSG:3006", allow_override=True)
    return gdf

def extract_year_from_nivadatum(series):
    # nivadatum like 20001009 (YYYYMMDD)
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[^0-9]", "", regex=True)
    year = pd.to_numeric(s.str.slice(0, 4), errors="coerce").astype("Int64")
    return year

def summarize_wells_by_catchment(gpkg_path, cat_shp_path, catchment_id, layer=WELLS_LAYER):
    catch = load_catchment(cat_shp_path, catchment_id)
    wells = load_wells(gpkg_path, layer=layer)

    # Align CRS and spatial filter
    wells_proj = wells.to_crs(catch.crs)
    inside = gpd.sjoin(wells_proj, catch[["ID_BSDB", "geometry"]], predicate="within", how="inner")
    if inside.empty:
        per_year_full = pd.DataFrame({"year": list(range(2000, 2025 + 1)), "n_wells": [0]*26})
        summary = {
            "most_wells_years": [],
            "max_wells": 0,
            "least_wells_years": list(per_year_full["year"].tolist()),
            "min_wells": 0,
            "timeseries_2000_2025": {int(r.year): int(r.n_wells) for r in per_year_full.itertuples(index=False)}
        }
        return per_year_full, summary

    # Year from nivadatum
    cols = {c.lower(): c for c in inside.columns}
    niv_col = cols.get("nivadatum")
    if niv_col is None:
        raise ValueError("Column 'nivadatum' not found in wells data.")

    inside = inside.copy()
    inside["year"] = extract_year_from_nivadatum(inside[niv_col])
    inside = inside.dropna(subset=["year"])
    inside["year"] = inside["year"].astype(int)

    # Aggregate and zero-fill 2000-2025
    per_year = inside.groupby("year").size().reset_index(name="n_wells")
    idx = pd.DataFrame({"year": list(range(2000, 2025 + 1))})
    per_year_full = idx.merge(per_year, on="year", how="left").fillna({"n_wells": 0})
    per_year_full["n_wells"] = per_year_full["n_wells"].astype(int)

    max_n = per_year_full["n_wells"].max()
    min_n = per_year_full["n_wells"].min()
    most_years = per_year_full.loc[per_year_full["n_wells"] == max_n, "year"].tolist()
    least_years = per_year_full.loc[per_year_full["n_wells"] == min_n, "year"].tolist()

    summary = {
        "most_wells_years": [int(x) for x in most_years],
        "max_wells": int(max_n),
        "least_wells_years": [int(x) for x in least_years],
        "min_wells": int(min_n),
        "timeseries_2000_2025": {int(r.year): int(r.n_wells) for r in per_year_full.itertuples(index=False)}
    }
    return per_year_full, summary

# ---------- Main ----------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Part A: recharge overall
    try:
        per_year_rech, rech_summary = summarize_recharge_overall(CSV_RECHARGE)
        print("\n=== Recharge (overall) ===")
        print(f"Wettest year(s): {rech_summary['wettest_years']} (mean={rech_summary['wettest_mean_recharge']:.2f} mm)")
        print(f"5 closest-to-wettest years (excluding wettest): {rech_summary['closest_to_wettest_years']}")
        print(f"Driest year(s): {rech_summary['driest_years']} (mean={rech_summary['driest_mean_recharge']:.2f} mm)")
        print(f"5 closest-to-driest years (excluding driest): {rech_summary['closest_to_driest_years']}")
        print(f"5 closest-to-median years: {rech_summary['closest_to_median_years']} "
              f"(median of yearly means={rech_summary['overall_median_of_yearly_means']:.2f} mm)")
        print("\nPer-year stats (year, n_points, mean_recharge):")
        print(per_year_rech[["year", "n_points", "mean_recharge"]].to_string(index=False))

        out_csv = os.path.join(OUTPUT_DIR, "recharge_per_year_overall.csv")
        per_year_rech.to_csv(out_csv, index=False)
        with open(os.path.join(OUTPUT_DIR, "recharge_summary.json"), "w") as f:
            json.dump(rech_summary, f, indent=2)
        print(f"\nSaved: {out_csv} and recharge_summary.json")
    except Exception as e:
        print(f"[Recharge] Error: {e}", file=sys.stderr)

    # Part B: wells by catchment
    try:
        cats = gpd.read_file(SHP_CATCHMENTS)
        if "ID_BSDB" not in cats.columns:
            raise ValueError("Catchment shapefile must contain 'ID_BSDB'.")
        catchment_id = prompt_catchment_id(cats)

        wells_per_year, wells_summary = summarize_wells_by_catchment(
            GPKG_WELLS, SHP_CATCHMENTS, catchment_id, layer=WELLS_LAYER
        )
        print("\n=== Wells in catchment ===")
        print(f"Most wells year(s): {wells_summary['most_wells_years']} (max={wells_summary['max_wells']})")
        print(f"Least wells year(s): {wells_summary['least_wells_years']} (min={wells_summary['min_wells']})")
        print("\nCounts per year 2000–2025:")
        print(wells_per_year.to_string(index=False))
        out_csv2 = os.path.join(OUTPUT_DIR, f"wells_per_year_ID_{catchment_id}.csv")
        wells_per_year.to_csv(out_csv2, index=False)
        with open(os.path.join(OUTPUT_DIR, f"wells_summary_ID_{catchment_id}.json"), "w") as f:
            json.dump(wells_summary, f, indent=2)
        print(f"\nSaved: {out_csv2} and wells_summary_ID_{catchment_id}.json")
    except Exception as e:
        print(f"[Wells] Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
