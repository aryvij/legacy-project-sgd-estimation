#!/usr/bin/env python3
# interface_main_sgd.py — single-screen, styled (narrow left pane, wide right, 2-per-row inputs)

import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# --- your app code ---
from core import main_sgd
from core.main_sgd import get_mean_discharge, is_coastal
from core.modflow_setup import setup_and_run_modflow

ROOT   = pathlib.Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
INPUT  = DATA / "input"
OUTPUT = DATA / "output"

LOGO_PATH = str(ROOT / "logo.png")

# keep GDAL memory modest
os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("RASTERIO_MAXIMUM_RAM", "512MB")

st.set_page_config(page_title="Coastal SGD", layout="wide", page_icon="🌊")

# ---------- helpers ----------
def run_workspace(catchment_id: int) -> pathlib.Path:
    return OUTPUT / "model_runs" / f"mf6_{catchment_id}"

def last_volume_budget_block(lst_text: str) -> str:
    """
    Return the *entire* final 'VOLUME BUDGET FOR ENTIRE MODEL ...' section
    from the MF6 .lst file. We return from that header to EOF so nothing
    gets truncated by blank lines or separators.
    """
    lines = lst_text.splitlines()
    hits = [i for i, L in enumerate(lines) if "VOLUME BUDGET FOR ENTIRE MODEL" in L.upper()]
    if not hits:
        return "(No 'VOLUME BUDGET FOR ENTIRE MODEL' section found.)"
    i0 = hits[-1]                 # last occurrence
    block = "\n".join(lines[i0:]) # take to end-of-file (safest)
    return block.strip("\n")

def read_lst_tail(cid: int) -> str:
    lst = run_workspace(cid) / f"gwf_{cid}.lst"
    if not lst.exists():
        return f"❌ Output file not found: {lst}"
    txt = lst.read_text(encoding="utf-8", errors="ignore")
    return last_volume_budget_block(txt)

def plot_heads(heads: np.ndarray, cid: int, year: int):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    vmin = np.nanmin(heads); vmax = np.nanmax(heads)
    im = ax.imshow(heads, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    fig.colorbar(im, ax=ax, label="Simulated Head (m)")
    ax.set_title(f"Catchment {cid} — Final Head (Top Layer) — {year}")
    return fig

def save_upload(upl, subdir, filename_hint) -> pathlib.Path | None:
    if not upl:
        return None
    folder = OUTPUT / "uploaded" / subdir
    folder.mkdir(parents=True, exist_ok=True)
    name = upl.name if getattr(upl, "name", None) else filename_hint
    p = folder / name
    with open(p, "wb") as f:
        f.write(upl.getbuffer())
    return p

def build_filepaths(
    year: int,
    rch_override: pathlib.Path | None,
    wells_gpkg: pathlib.Path | None,
    rivers_zip: pathlib.Path | None,
    lakes_zip: pathlib.Path | None,
    sea_csv: pathlib.Path | None
) -> dict:
    fp = {
        "dem"         : INPUT / "dem" / "elevation_sweden.tif",
        "catchment"   : main_sgd.CATCH_SHP,
        "recharge"    : OUTPUT / "recharge_yearly" / f"recharge_egdi_gldas_{year}.tif",
        "soil_perm"   : main_sgd.SOIL_PERM,
        "soil_depth"  : INPUT / "aquifer_data" / "jorddjupsmodell" / "jorddjupsmodell_10x10m.tif",
        "conductivity": main_sgd.BEDROCK_K,
        "sea_level"   : INPUT / "sea_level" / "yearly_average_sea_level.csv",
        "coast"       : main_sgd.COAST_FOR_MF,
        "wells"       : INPUT / "well_data" / "brunnar.gpkg",
        "rivers"      : main_sgd.RIVERS_SHP,
        "lakes"       : main_sgd.LAKES_SHP,
        "output"      : OUTPUT,
    }
    if rch_override: fp["recharge"] = rch_override
    if sea_csv:      fp["sea_level"] = sea_csv
    if wells_gpkg:   fp["wells"]  = wells_gpkg
    if rivers_zip:   fp["rivers"] = rivers_zip
    if lakes_zip:    fp["lakes"]  = lakes_zip
    return {k: str(v) for k, v in fp.items()}

# ---------- CSS (compact top, small logo, column gap) ----------
st.markdown("""
<style>
/* Give the main content some breathing room from the very top */
.stMain > div.block-container { padding-top: 2.75rem; }

/* Headline styling (no underline) */
h1 {
  text-align: center;
  margin: 0 0 1.25rem 0;
  font-size: 2.2rem;
  font-weight: 700;
  color: #1e3a8a;
  line-height: 1.2;
}
h1 + hr, hr { display: none; border: 0; height: 0; }

/* Slight separation between left/right columns (newer Streamlit class) */
.st-ae.st-af .stColumn { gap: 1.25rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Header: small logo left, big title right (centered) ----------
hdr_l, hdr_r = st.columns([0.25, 1.75])
with hdr_l:
    if pathlib.Path(LOGO_PATH).exists():
        st.image(LOGO_PATH, output_format="PNG", use_container_width=False, clamp=True)
with hdr_r:
    st.markdown("# Coastal SGD — interactive run")

# ---------- Main: left = inputs/uploaders, right = run setup + results ----------
left, right = st.columns([0.9, 1.45], gap="large")

with left:
    st.markdown("#### Inputs")

    # multipliers (two per row)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        soilK = st.number_input("Soil K ×", value=1.4528859067902082)
    with r1c2:
        rockK = st.number_input("Rock K ×", value=2.9111852425870564)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        rivM  = st.number_input("River cond ×", value=1.0624105361822733)
    with r2c2:
        ghbM  = st.number_input("GHB cond ×",  value=0.843504158508959)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        rchM  = st.number_input("Recharge ×",  value=0.9068883707497266)
    with r3c2:
        st.write("")  # spacer to keep grid tidy

    st.divider()
    st.caption("Upload optional replacements")

    u1, u2 = st.columns(2)
    with u1:
        rch_upl   = st.file_uploader("Recharge GeoTIFF", type=["tif","tiff"], key="upl_rch")
    with u2:
        wells_upl = st.file_uploader("Wells (GPKG)", type=["gpkg"], key="upl_wells")

    u3, u4 = st.columns(2)
    with u3:
        rivers_upl= st.file_uploader("Rivers (zipped SHP)", type=["zip"], key="upl_riv")
    with u4:
        lakes_upl = st.file_uploader("Lakes (zipped SHP)",  type=["zip"], key="upl_lakes")

    sea_upl   = st.file_uploader("Sea-level CSV (optional)", type=["csv"], key="upl_sea")

with right:
    st.markdown("### Run setup")
    rs1, rs2, rs3 = st.columns([1,1,1])
    with rs1:
        cid  = st.number_input("Catchment ID", min_value=1, step=1, value=204)
    with rs2:
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2019, step=1)
    with rs3:
        coastal_buffer = st.number_input("Coastal buffer (m)", min_value=0.0, step=50.0, value=200.0)

    run = st.button("Run workflow", type="primary")

    if run:
        with st.spinner("Running…"):
            q, status = get_mean_discharge(int(cid))
            st.write(f"**Mean-annual discharge Q** = {q:.2f} m³ s⁻¹ ({status})")

            if not is_coastal(int(cid)):
                st.info("Inland basin — no SGD simulation required.")
            else:
                # Save uploads (if any)
                rch_path   = save_upload(rch_upl,   "recharge", "recharge.tif")
                wells_path = save_upload(wells_upl, "wells",    "wells.gpkg")
                rivers_path= save_upload(rivers_upl,"rivers",   "rivers.zip")
                lakes_path = save_upload(lakes_upl, "lakes",    "lakes.zip")
                sea_path   = save_upload(sea_upl,   "sea_level","sea.csv")

                # Ensure default recharge exists if none uploaded
                if rch_path is None:
                    default_rch = OUTPUT / "recharge_yearly" / f"recharge_egdi_gldas_{int(year)}.tif"
                    if not default_rch.exists():
                        st.error(f"Recharge raster missing: {default_rch}")
                        st.stop()

                filepaths = build_filepaths(int(year), rch_path, wells_path, rivers_path, lakes_path, sea_path)

                # internal MF6 path (hidden from UI)
                mf6_exe_path = r"C:\Users\aryapv\AppData\Local\Programs\mf6.6.1_win64\bin\mf6.exe"

                try:
                    heads, dem_tr, dem_crs, catch_poly = setup_and_run_modflow(
                        catchment_id        = int(cid),
                        filepaths           = filepaths,
                        coastal_buffer      = float(coastal_buffer),
                        mf6_exe             = mf6_exe_path,
                        recharge_year       = int(year),
                        soilK_multiplier    = float(soilK),
                        rockK_multiplier    = float(rockK),
                        riv_cond_multiplier = float(rivM),
                        ghb_cond_multiplier = float(ghbM),
                        rch_multiplier      = float(rchM),
                    )
                except Exception as e:
                    st.error(f"Run failed: {e}")
                    st.stop()

        # ---- Post-run UI ----
        if heads is None:
            st.error("No heads returned.")
        else:
            st.success("MODFLOW simulation completed.")
            st.pyplot(plot_heads(heads, int(cid), int(year)), use_container_width=True)

            # Read the full, final budget block once
            budget = read_lst_tail(int(cid))

            # Show it in an expanded panel (single place)
            with st.expander("Volume budget (last block)", expanded=True):
                st.code(budget, language="text")

            # Short, professional report (includes full budget)
            report = (
                f"Catchment {cid}\n"
                f"Year {year}\n"
                f"Q = {q:.2f} m³ s⁻¹ ({status})\n\n"
                f"{budget}\n"
            )
            st.download_button(
                "Download report",
                data=report,
                file_name=f"report_{cid}_{year}.txt",
                mime="text/plain",
            )
