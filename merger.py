import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "astrobit_secondary.csv")
OUT_FILE = os.path.join(BASE_DIR, "astrobit_submission.csv")

BASELINE_DAYS = 1460

df = pd.read_csv(INPUT_FILE)

df["Rp_Rs"] = np.sqrt(df["Depth_ppm"] / 1e6)
df["n_transits"] = np.floor(BASELINE_DAYS / df["Period_days"]).astype(int)

df = df[[
    "KIC",
    "Confidence_%",
    "Period_days",
    "Depth_ppm",
    "Duration_hours",
    "Rp_Rs",
    "n_transits"
]]

df.to_csv(OUT_FILE, index=False)

print(f"Saved: {OUT_FILE}")
