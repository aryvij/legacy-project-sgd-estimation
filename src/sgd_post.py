# sgd_post.py
import os
import csv
import numpy as np
import flopy

def extract_sgd_from_cbc(base_ws: str,
                         catchment: int,
                         year: int | None,
                         out_csv: str | None = None) -> dict:
    """
    Read gwf_<catchment>.cbc, sum coastal GHB outflow at the last timestep,
    and optionally append to a CSV. Returns a dict with the totals.

    Interpretation:
      - MODFLOW-6 budget convention: q > 0 is flow *into* the model,
        q < 0 is flow *out of* the model. For SGD we want coastal outflow,
        so we sum abs(q) for all GHB records with q < 0 at the final time.
    """
    cbc_path = os.path.join(base_ws, f"gwf_{catchment}.cbc")
    if not os.path.exists(cbc_path):
        raise FileNotFoundError(f"Cell-by-cell budget not found: {cbc_path}")

    # Open with double precision to match your budget summary code
    cbc = flopy.utils.CellBudgetFile(cbc_path, precision='double')

    # Find last (final) time in the CBC file
    times = cbc.get_times()
    if not times:
        raise RuntimeError("No times found in CBC.")
    t_last = times[-1]

    # Get all GHB records at the last time step
    ghb = cbc.get_data(text="GHB", totim=t_last)
    if not ghb:
        # Some builds store text as bytes; try that too
        ghb = cbc.get_data(text=b"GHB", totim=t_last)
    if not ghb:
        # No GHB flows recorded
        ghb_out_m3d = 0.0
    else:
        # ghb is usually a list with one np.recarray of dtype (..., 'q')
        arr = ghb[-1]
        if isinstance(arr, np.ndarray) and ('q' in (arr.dtype.names or ())):
            q = arr['q'].astype(float)
        else:
            # Fallback: try to interpret as plain array
            q = np.array(arr, dtype=float).ravel()

        # Sum outflows (q < 0) and take magnitude
        ghb_out_m3d = float(np.abs(np.nansum(q[q < 0])))

    result = {
        "catchment": int(catchment),
        "year": int(year) if year is not None else None,
        "sgd_m3_per_day": ghb_out_m3d,
        "sgd_m3_per_year": ghb_out_m3d * 365.0
    }

    # Optional: append to CSV
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["catchment","year","sgd_m3_per_day","sgd_m3_per_year","cbc_path"])
            w.writerow([result["catchment"], result["year"],
                        f"{result['sgd_m3_per_day']:.6e}",
                        f"{result['sgd_m3_per_year']:.6e}",
                        cbc_path])
    return result
