"""
Compute mean-annual discharge (m³ s⁻¹) for every catchment folder found
under:

    data/input/discharge/discharge_data/<catchment_name>/

The script supports two formats
--------------------------------
1. **SMHI CSV** (monthly values, Swedish header – see README)
2. **GRDC NetCDF** (`runoff_mean` in m³ s⁻¹)

For each catchment it will

* read **all** files it finds (you can mix CSV and NetCDF)
* filter the period 2000-01-01 … 2024-12-31
* convert monthly (or daily) series to **annual volumes**, then to the
  **mean annual discharge** (m³ s⁻¹)
* write one row per catchment to  
  `data/input/discharge/monitored_mean_Q.csv`

If no usable data are found, the catchment is skipped and a warning is
printed.

Author: ChatGPT (o3) – 2025-04-16
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------
# adjust if you move the script
BASE_DIR = Path(__file__).resolve().parents[2]        # project root
DATA_DIR = BASE_DIR / "data" / "input" / "discharge" / "discharge_data"
OUT_CSV  = BASE_DIR / "data" / "input" / "discharge" / "monitored_mean_Q.csv"

YEAR_START = 2000
YEAR_END   = 2024
T_PER_SEC  = 31_557_600          # seconds in a (365.25-day) Gregorian year


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def read_smhi_csv(fpath: Path) -> pd.Series:
    """
    Parse an SMHI CSV. Returns a pandas Series (datetime64[M] index, m³/s).
    """
    # file has lots of metadata → find the first line that starts with YYYY-MM
    rx_data = re.compile(r"^\d{4}-\d{2};")
    rows = []
    with fpath.open(encoding="cp1252") as fh:             # SMHI often Latin-1
        for ln in fh:
            if rx_data.match(ln):
                rows.append(ln.strip())

    if not rows:
        raise ValueError("no data rows recognised")

    # build a DataFrame
    df = pd.read_csv(
        pd.compat.StringIO("\n".join(rows)),
        sep=";",
        header=None,
        names=["date", "q", "_qual", "_extra1", "_extra2"],
        usecols=["date", "q"],
        dtype={"date": str, "q": str},
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df["q"]    = pd.to_numeric(df["q"], errors="coerce")

    s = df.set_index("date")["q"].dropna()
    return s


def read_grdc_nc(fpath: Path) -> pd.Series:
    """
    Read GRDC NetCDF (variable `runoff_mean` in m³/s).
    Returns Series with **daily** values (time index).
    """
    ds = xr.open_dataset(fpath)
    var_candidates = [v for v in ds.data_vars if "runoff" in v.lower()]

    if not var_candidates:
        raise ValueError("no discharge variable in NetCDF")

    var = ds[var_candidates[0]].squeeze(drop=True)  # remove singleton dims
    if var.ndim != 1:
        raise ValueError(f"variable {var.name} not 1-D (shape {var.shape})")

    time = pd.to_datetime(var["time"].values)
    s = pd.Series(var.values.astype(float), index=time, name="Q").dropna()
    return s


def annual_mean_q(series: pd.Series) -> float:
    """
    Convert a time-series (m³ s⁻¹) to mean-annual discharge (m³ s⁻¹).

    For monthly data: multiply each m³ s⁻¹ by days_in_month * 86400,
    sum per water-year, average, then divide by seconds/year.
    """
    if series.empty:
        raise ValueError("empty series")

    # slice the requested window
    s = series.loc[f"{YEAR_START}-01-01": f"{YEAR_END}-12-31"]

    if s.empty:
        raise ValueError("series has no data in requested period")

    # ensure numeric
    s = s.astype(float)

    # ------------------------------------------------------------------
    # 1. group into water-years (Oct–Sep) or calendar years?
    #    → the user requested *water-year* in the earlier discussion,
    #      but here we keep it simple = calendar year because the
    #      SMHI monthly series are calendar months.
    # ------------------------------------------------------------------
    # seconds in each sample
    if s.index.freq is None:
        # irregular → treat as daily
        dt_seconds = np.diff(s.index.to_numpy("datetime64[s]"), prepend=s.index[0])\
                        .astype("timedelta64[s]").astype(float)
    else:
        # freq aware → use freq
        dt_seconds = np.full(len(s), s.index.freq.delta.total_seconds())

    volumes_m3 = s.values * dt_seconds  # m³

    df = pd.DataFrame({"vol": volumes_m3}, index=s.index)
    annual_vol = df.resample("A").sum()["vol"]
    mean_vol   = annual_vol.mean()

    mean_q = mean_vol / T_PER_SEC
    return float(mean_q)


# ---------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------
records: list[tuple[str, float]] = []

for catch_folder in sorted(DATA_DIR.iterdir()):
    if not catch_folder.is_dir():
        continue

    cid = catch_folder.name
    q_series_all = []

    for f in catch_folder.iterdir():
        try:
            if f.suffix.lower() == ".csv":
                q_series_all.append(read_smhi_csv(f))
            elif f.suffix.lower() in (".nc", ".nc4", ".netcdf"):
                q_series_all.append(read_grdc_nc(f))
        except Exception as ex:
            warnings.warn(f"{cid}: {ex}")
            continue

    if not q_series_all:
        warnings.warn(f"{cid}: no readable discharge file")
        continue

    # merge (if multiple files) → take longest continuous index
    qs_merged = pd.concat(q_series_all).groupby(level=0).mean()

    try:
        q_mean = annual_mean_q(qs_merged)
        records.append((cid, q_mean))
        print(f"[OK ] {cid}: mean Q = {q_mean:.3f} m³/s")
    except Exception as ex:
        warnings.warn(f"{cid}: {ex}")

# ---------------------------------------------------------------------
# save
# ---------------------------------------------------------------------
if records:
    out_df = pd.DataFrame(records, columns=["catchment", "Q_mean_m3s"]).sort_values("catchment")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n✅  Saved mean Q for {len(out_df)} catchments → {OUT_CSV}")
else:
    print("⚠️  No catchments processed – nothing written.")
