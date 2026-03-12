#!/usr/bin/env python3
# plot_uncertainty_violin.py
# Dual-panel violin plot: RMSE (2018 vs 2019) and SGD (2018 vs 2019)

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_metric(csv_path, colname):
    df = pd.read_csv(csv_path)
    if colname not in df.columns:
        raise ValueError(f"Column '{colname}' not found in {csv_path}. Columns: {list(df.columns)}")
    v = df[colname].astype(float).values
    v = v[np.isfinite(v)]
    return v


def draw_violin(ax, data_by_year, title, ylabel):
    """
    data_by_year: list of (label, 1D array)
    """
    labels = [lab for lab, _ in data_by_year]
    arrays = [arr for _, arr in data_by_year]

    # Violin bodies — light grey fill, black edge
    parts = ax.violinplot(arrays, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("lightgrey")
        pc.set_alpha(0.8)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.8)

    # Add black median dot + 5th–95th whiskers
    for i, arr in enumerate(arrays, start=1):
        if arr.size == 0:
            continue
        p5, med, p95 = np.percentile(arr, [5, 50, 95])
        ax.plot([i-0.25, i+0.25], [p5, p5], lw=1.2, color="black")
        ax.plot([i-0.25, i+0.25], [p95, p95], lw=1.2, color="black")
        ax.plot(i, med, "o", color="black", ms=5)

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels)
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv2018", required=True)
    ap.add_argument("--csv2019", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--outname", default="uncertainty_violin_2018_2019.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, args.outname)

    # Load metrics
    rmse18 = load_metric(args.csv2018, "RMSE")
    rmse19 = load_metric(args.csv2019, "RMSE")
    sgd18  = load_metric(args.csv2018, "SGD_m3d")
    sgd19  = load_metric(args.csv2019, "SGD_m3d")

    # Figure
        # --- Adjust plotting ---
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)

    # RMSE subplot
    draw_violin(
        axs[0],
        [("2018 (dry)", rmse18), ("2019 (wet)", rmse19)],
        title="RMSE distribution",
        ylabel="RMSE (m)",
    )

    # SGD subplot — rescale to 10³ m³/d
    sgd18_k = sgd18 / 1000.0
    sgd19_k = sgd19 / 1000.0
    draw_violin(
        axs[1],
        [("2018 (dry)", sgd18_k), ("2019 (wet)", sgd19_k)],
        title="SGD distribution",
        ylabel="SGD (×10³ m³ d⁻¹)",
    )

    # Common legend
    med_proxy, = axs[0].plot([], [], "o", color="black", label="Median", ms=5)
    p_proxy,  = axs[0].plot([], [], "-", color="black", lw=1.2, label="5th–95th percentile")
    # Move legend a bit lower (was -0.02)
    fig.legend(handles=[med_proxy, p_proxy],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08), frameon=False)

    plt.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    print(f"[ok] wrote: {outpath}")

if __name__ == "__main__":
    main()
