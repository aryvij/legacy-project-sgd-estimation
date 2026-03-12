#!/usr/bin/env python3
## activate
#.\env\Scripts\Activate.ps1

# make the dual-panel figure, hide ghb explicitly (and auto-drop tiny effects < 0.02)
#python -u src\plot_sobol_dual.py `
#  --json2018 "C:\Users\aryapv\OneDrive - KTH\Modelling_SGD_Arya\SGD_model\data\output\sensitivity_analysis\sobol_204_2018_n32_with_sgd\sobol_indices_c204_y2018.json" `
#  --json2019 "C:\Users\aryapv\OneDrive - KTH\Modelling_SGD_Arya\SGD_model\data\output\sensitivity_analysis\sobol_204_2019_n32_with_sgd\sobol_indices_c204_y2019.json" `
#  --hide ghb `
#  --auto-threshold 0.02 `
#  --out "C:\Users\aryapv\OneDrive - KTH\Modelling_SGD_Arya\SGD_model\data\output\figures\sobol_2018_2019_rmse_sgd.png" `
#  --dpi 600

#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
import json, argparse, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch

def load_indices(fp):
    with open(fp, "r") as f:
        d = json.load(f)
    out = {}
    for key in ("RMSE", "SGD"):
        if key in d:
            out[key] = {
                "names": d[key]["names"],
                "S1": np.array(d[key]["S1"], dtype=float),
                "ST": np.array(d[key]["ST"], dtype=float),
            }
    return out

def filter_params(names, S1_a, ST_a, S1_b, ST_b, hide, thr):
    keep = []
    for i, n in enumerate(names):
        if n in hide:
            continue
        if thr is not None:
            if max(abs(S1_a[i]), abs(ST_a[i]), abs(S1_b[i]), abs(ST_b[i])) < thr:
                continue
        keep.append(i)
    names_f = [names[i] for i in keep]
    return (names_f, S1_a[keep], ST_a[keep], S1_b[keep], ST_b[keep])

def make_panel(ax, names, S1_2018, ST_2018, S1_2019, ST_2019,
               title, bar_w=0.18, fs_tick=12, fs_title=14, show_values=True):

    # Soft cohesive colors (cool tones)
    c2018_s1 = "#87AFC7"  # muted steel blue
    c2018_st = "#A7C7E7"  # lighter blue
    c2019_s1 = "#98C39B"  # soft green
    c2019_st = "#B7D6B9"  # pale green

    x = np.arange(len(names))
    offsets = (-1.5, -0.5, +0.5, +1.5)

    bars = []
    bars += ax.bar(x + offsets[0]*bar_w, S1_2018, width=bar_w, color=c2018_s1, label="2018 S1")
    bars += ax.bar(x + offsets[1]*bar_w, S1_2019, width=bar_w, color=c2019_s1, label="2019 S1")
    bars += ax.bar(x + offsets[2]*bar_w, ST_2018, width=bar_w, color=c2018_st, label="2018 ST")
    bars += ax.bar(x + offsets[3]*bar_w, ST_2019, width=bar_w, color=c2019_st, label="2019 ST")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=fs_tick)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.set_title(title, fontsize=fs_title)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.margins(y=0.1)

    # Optional numeric labels
    if show_values:
        for bar in bars:
            h = bar.get_height()
            if not np.isfinite(h): 
                continue
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=16, color="#333")

    # Legend handles
    handles = [
        Patch(facecolor=c2018_s1, label="2018 S1"),
        Patch(facecolor=c2019_s1, label="2019 S1"),
        Patch(facecolor=c2018_st, label="2018 ST"),
        Patch(facecolor=c2019_st, label="2019 ST"),
    ]
    return handles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json2018", required=True)
    ap.add_argument("--json2019", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--hide", default="", help="Comma-separated names to hide (e.g. ghb)")
    ap.add_argument("--auto-threshold", type=float, default=None)
    ap.add_argument("--fontsize", type=int, default=16)
    ap.add_argument("--title-size", type=int, default=16)
    ap.add_argument("--legend-size", type=int, default=16)
    ap.add_argument("--no-values", action="store_true")
    args = ap.parse_args()

    hide = [s.strip() for s in args.hide.split(",") if s.strip()]
    A = load_indices(args.json2018)
    B = load_indices(args.json2019)
    have_sgd = ("SGD" in A) and ("SGD" in B)

    plt.rcParams.update({
        "font.size": args.fontsize,
        "axes.labelsize": args.fontsize,
        "xtick.labelsize": args.fontsize,
        "ytick.labelsize": args.fontsize,
        "axes.titlesize": args.title_size,
    })

    ncols = 2 if have_sgd else 1
    fig, axes = plt.subplots(1, ncols, figsize=(12, 5), sharey=False)
    if ncols == 1:
        axes = [axes]

    # Panel 1: RMSE
    names = A["RMSE"]["names"]
    (names_rmse, S1_18f, ST_18f, S1_19f, ST_19f) = filter_params(
        names, A["RMSE"]["S1"], A["RMSE"]["ST"], B["RMSE"]["S1"], B["RMSE"]["ST"], hide, args.auto_threshold)

    h_rmse = make_panel(axes[0], names_rmse, S1_18f, ST_18f, S1_19f, ST_19f,
                        "Global sensitivity (RMSE)",
                        fs_tick=args.fontsize, fs_title=args.title_size,
                        show_values=not args.no_values)
    axes[0].set_ylabel("Sobol index")

    # Panel 2: SGD
    if have_sgd:
        names = A["SGD"]["names"]
        (names_sgd, S1_18f, ST_18f, S1_19f, ST_19f) = filter_params(
            names, A["SGD"]["S1"], A["SGD"]["ST"], B["SGD"]["S1"], B["SGD"]["ST"], hide, args.auto_threshold)
        make_panel(axes[1], names_sgd, S1_18f, ST_18f, S1_19f, ST_19f,
                   "Global sensitivity (SGD)",
                   fs_tick=args.fontsize, fs_title=args.title_size,
                   show_values=not args.no_values)

    # Shared legend below both plots
    fig.legend(h_rmse, ["2018 S1", "2019 S1", "2018 ST", "2019 ST"],
               loc="lower center", ncol=4, fontsize=args.legend_size,
               bbox_to_anchor=(0.5, -0.05), frameon=False)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[ok] saved -> {args.out}")

if __name__ == "__main__":
    main()
