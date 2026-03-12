#This script is to clean the RMSE convergence plot for year 2020 - calibration and the ouput is used in artyicle
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Paths ===
csv_path = r"C:\Users\aryapv\OneDrive - KTH\Modelling_SGD_Arya\SGD_model\data\output\test_run_8_2010\calib_trace_c204_y2010.csv"
out_dir = r"C:\Users\aryapv\OneDrive - KTH\Modelling_SGD_Arya\SGD_model\data\output\test_run_8_2010\for_article"
os.makedirs(out_dir, exist_ok=True)

# === Load data ===
df = pd.read_csv(csv_path)

# Filter out penalty evaluations (RMSE = 1e6)
df_clean = df[df["rmse"] < 1e5].copy()

# === Plot ===
plt.figure(figsize=(7, 4.5))

plt.scatter(df_clean["iter"], df_clean["rmse"],
            s=45, color="#1f77b4", alpha=0.9)

plt.xlabel("Iteration", fontsize=13)
plt.ylabel("RMSE (m)", fontsize=13)
plt.title("Calibration Convergence — Catchment 204 (2010)",
          fontsize=14, fontweight="bold", pad=12)

# === Style ===
plt.grid(False)  # remove background grid lines
plt.tick_params(axis='both', which='major', labelsize=11)

# Make axes crisp
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(0.8)

plt.tight_layout()

# === Save ===
out_file = os.path.join(out_dir, "fig_calibration_convergence_c204_y2010.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"✅ Figure saved to:\n{out_file}")
