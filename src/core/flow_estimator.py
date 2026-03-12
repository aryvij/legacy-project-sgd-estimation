# flow_estimator.py
# 2025-09-30 Arya Vijayan
import pathlib, pandas as pd, geopandas as gpd

ROOT   = pathlib.Path(__file__).resolve().parents[2]
CSV_Q  = ROOT/"data"/"input"/"discharge"/"monitored_mean_Q.csv"
CATCH  = ROOT/"data"/"input"/"shapefiles"/"catchment"/"bsdbs.shp"

_mon_q_df = (
    pd.read_csv(CSV_Q, sep=r"[;,]", engine="python")
      .rename(columns=str.strip)
)
_mon_q_df["ID_BSDB"] = _mon_q_df["ID_BSDB"].astype(str)
_cg = gpd.read_file(CATCH)[["ID_BSDB","AREA_KM2"]]
_cg["ID_BSDB"] = _cg["ID_BSDB"].astype(str)

def get_mean_discharge(catchment_id: str|int) -> tuple[float,str]:
    cid = str(catchment_id)
    row = _mon_q_df[_mon_q_df.ID_BSDB==cid]
    if not row.empty:
        return float(row.Q_mean_m3s.iloc[0]), "MONITORED"
    crow = _cg[_cg.ID_BSDB==cid]
    if crow.empty:
        raise ValueError(f"{cid} not in catchment shapefile")
    A = float(crow.AREA_KM2.iloc[0])
    return 0.0030 * A**1.15, "UNMONITORED_QA"
