import pandas as pd
import numpy as np
import glob
import os

RESULTS_FILE = r"C:\Users\LSESH\OneDrive\Desktop\prjs\Astrobit\astrobit_results.csv"
LIGHTCURVE_DIR = r"C:\Users\LSESH\OneDrive\Desktop\prjs\Astrobit\private"
OUTPUT_FILE = r"C:\Users\LSESH\OneDrive\Desktop\prjs\Astrobit\astrobit_submission.csv"

df = pd.read_csv(RESULTS_FILE)

# Planet-to-star radius ratio from transit depth
df["Rp_Rs"] = np.sqrt(df["Depth_ppm"] / 1_000_000)

def get_baseline_days(kic):
    files = glob.glob(os.path.join(LIGHTCURVE_DIR, f"*{kic}*"))
    if not files:
        return np.nan

    try:
        file = files[0]
        data = pd.read_parquet(file)

        if "time" not in data.columns:
            return np.nan

        t = pd.to_numeric(data["time"], errors="coerce").dropna()

        if len(t) < 2:
            return np.nan

        return t.max() - t.min()

    except Exception:
        return np.nan

# Calculate transit count
df["baseline_days"] = df["KIC"].apply(get_baseline_days)

df["n_transits"] = np.floor(
    df["baseline_days"] / df["Period_days"]
).astype("Int64")

# Final submission columns
submission = df[
    [
        "KIC",
        "Confidence_%",
        "Period_days",
        "Depth_ppm",
        "Duration_hours",
        "Rp_Rs",
        "n_transits"
    ]
].copy()

submission.to_csv(OUTPUT_FILE, index=False)

print("\nFINAL SUBMISSION:")
print(submission.to_string(index=False))

print("\nSaved to:")
print(OUTPUT_FILE)