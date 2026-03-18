# flow_estimator.py
# 2025-09-30 Arya Vijayan
import pathlib, pandas as pd, geopandas as gpd

_DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = _DEFAULT_ROOT / "data" / "input"

# Lazy-loaded module-level cache
_mon_q_df = None
_cg = None
_loaded_root = None

def _ensure_loaded(data_root=None):
    global _mon_q_df, _cg, _loaded_root
    if data_root is None:
        data_root = _DEFAULT_INPUT
    else:
        data_root = pathlib.Path(data_root)
    if _mon_q_df is not None and _loaded_root == data_root:
        return
    csv_q = data_root / "discharge" / "monitored_mean_Q.csv"
    catch = data_root / "shapefiles" / "catchment" / "bsdbs.shp"
    _mon_q_df = (
        pd.read_csv(csv_q, sep=r"[;,]", engine="python")
          .rename(columns=str.strip)
    )
    _mon_q_df["ID_BSDB"] = _mon_q_df["ID_BSDB"].astype(str)
    _cg = gpd.read_file(catch)[["ID_BSDB","AREA_KM2"]]
    _cg["ID_BSDB"] = _cg["ID_BSDB"].astype(str)
    _loaded_root = data_root

def get_mean_discharge(catchment_id: str|int, data_root=None) -> tuple[float,str]:
    _ensure_loaded(data_root)
    cid = str(catchment_id)
    row = _mon_q_df[_mon_q_df.ID_BSDB==cid]
    if not row.empty:
        return float(row.Q_mean_m3s.iloc[0]), "MONITORED"
    crow = _cg[_cg.ID_BSDB==cid]
    if crow.empty:
        raise ValueError(f"{cid} not in catchment shapefile")
    A = float(crow.AREA_KM2.iloc[0])
    return 0.0030 * A**1.15, "UNMONITORED_QA"
