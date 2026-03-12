#!/usr/bin/env python3
"""
Plot OAT results (scatter + line) from a CSV produced by sensitivity_oat.py.
How to run:
python src\plot_oat_results.py `
  --csv "data\output\sensitivity_analysis\2019\sens_oat_c204_y2019.csv" `
  --outdir "data\output\sensitivity_analysis\2019"

Inputs
------
--csv      Path to the OAT CSV (e.g. data/output/sensitivity_analysis/2019/sens_oat_c204_y2019.csv)
--outdir   Folder to save figures (defaults to the CSV folder)

Outputs
-------
- oat_scatter_dRMSE.png     : delta_frac vs dRMSE (scatter), one color per param
- oat_line_dRMSE.png        : delta_frac vs dRMSE (line), one line per param
- oat_scatter_dSGD.png      : delta_frac vs dSGD_m3d (scatter, if column exists)
- oat_line_dSGD.png         : delta_frac vs dSGD_m3d (line, if column exists)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to sens_oat CSV file")
    ap.add_argument("--outdir", default=None, help="Output folder for figures (default: same as CSV)")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    outdir = args.outdir or os.path.dirname(csv_path)
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Basic sanity: keep expected columns if present
    expected_cols = ["param", "delta_frac", "dRMSE"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    # Keep only finite rows per metric
    df_rmse = df[np.isfinite(df["delta_frac"]) & np.isfinite(df["dRMSE"])].copy()

    # ---------- Scatter: delta vs dRMSE ----------
    plt.figure(figsize=(7,5))
    for p, g in df_rmse.groupby("param"):
        plt.scatter(g["delta_frac"], g["dRMSE"], label=p)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("delta_frac")
    plt.ylabel("dRMSE")
    plt.title("OAT: delta vs dRMSE (scatter)")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_scatter_rmse = os.path.join(outdir, "oat_scatter_dRMSE.png")
    plt.savefig(out_scatter_rmse, dpi=200)
    plt.close()
    print(f"saved {out_scatter_rmse}")

    # ---------- Line: delta vs dRMSE ----------
    # For a clean line, sort by delta within each param
    plt.figure(figsize=(7,5))
    for p, g in df_rmse.groupby("param"):
        g2 = g.sort_values("delta_frac")
        plt.plot(g2["delta_frac"], g2["dRMSE"], marker="o", label=p)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("delta_frac")
    plt.ylabel("dRMSE")
    plt.title("OAT: delta vs dRMSE (line)")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_line_rmse = os.path.join(outdir, "oat_line_dRMSE.png")
    plt.savefig(out_line_rmse, dpi=200)
    plt.close()
    print(f"saved {out_line_rmse}")

    # ---------- If SGD columns exist, plot those too ----------
    if "dSGD_m3d" in df.columns:
        df_sgd = df[np.isfinite(df["delta_frac"]) & np.isfinite(df["dSGD_m3d"])].copy()
        if not df_sgd.empty:
            # Scatter: delta vs dSGD
            plt.figure(figsize=(7,5))
            for p, g in df_sgd.groupby("param"):
                plt.scatter(g["delta_frac"], g["dSGD_m3d"], label=p)
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.xlabel("delta_frac")
            plt.ylabel("dSGD (m3/d)")
            plt.title("OAT: delta vs dSGD (scatter)")
            plt.legend(frameon=False)
            plt.tight_layout()
            out_scatter_sgd = os.path.join(outdir, "oat_scatter_dSGD.png")
            plt.savefig(out_scatter_sgd, dpi=200)
            plt.close()
            print(f"saved {out_scatter_sgd}")

            # Line: delta vs dSGD
            plt.figure(figsize=(7,5))
            for p, g in df_sgd.groupby("param"):
                g2 = g.sort_values("delta_frac")
                plt.plot(g2["delta_frac"], g2["dSGD_m3d"], marker="o", label=p)
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.xlabel("delta_frac")
            plt.ylabel("dSGD (m3/d)")
            plt.title("OAT: delta vs dSGD (line)")
            plt.legend(frameon=False)
            plt.tight_layout()
            out_line_sgd = os.path.join(outdir, "oat_line_dSGD.png")
            plt.savefig(out_line_sgd, dpi=200)
            plt.close()
            print(f"saved {out_line_sgd}")
        else:
            print("no finite dSGD_m3d values to plot; skipped SGD figures.")
    else:
        print("column dSGD_m3d not found; skipped SGD figures.")

if __name__ == "__main__":
    main()
