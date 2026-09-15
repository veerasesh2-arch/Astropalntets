
import pandas as pd
import numpy as np
import joblib

FILE = r"astrobit_tests.csv"
MODEL_FILE = r"astrobit_rf.pkl"
OUT_FILE = r"astrobit_results.csv"

FEATURES = [
    "Period_days",
    "Depth_ppm",
    "Duration_hours",
    "SDE"
]

df = pd.read_csv(FILE)

print("\n================ TEST DATA ================\n")
print(f"Stars : {len(df)}")
print(f"Columns : {df.columns.tolist()}")

df = df.replace([np.inf, -np.inf], np.nan)

missing = [c for c in FEATURES if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=FEATURES).copy()

X = df[FEATURES]

# ================= LOAD MODEL =================

model = joblib.load(MODEL_FILE)

# ================= PREDICTION =================

df["Confidence_%"] = model.predict_proba(X)[:, 1] * 100
df["Prediction"] = (df["Confidence_%"] >= 50).astype(int)

# ================= RADIUS RATIO =================

df["Rp_Rs_pred"] = np.sqrt(
    np.maximum(df["Depth_ppm"], 0) / 1_000_000
)

# ================= SORT =================

df = df.sort_values(
    "Confidence_%",
    ascending=False
).reset_index(drop=True)

# ================= NEAT TABLE =================

result = df[
    [
        "KIC",
        "Prediction",
        "Confidence_%",
        "Period_days",
        "Depth_ppm",
        "Duration_hours",
        "SDE",
        "Rp_Rs_pred"
    ]
].copy()

print("\n================ ALL TEST PREDICTIONS ================\n")

print(
    result.to_string(
        index=False,
        formatters={
            "Confidence_%": "{:.2f}".format,
            "Period_days": "{:.4f}".format,
            "Depth_ppm": "{:.2f}".format,
            "Duration_hours": "{:.2f}".format,
            "SDE": "{:.2f}".format,
            "Rp_Rs_pred": "{:.5f}".format
        }
    )
)

# ================= SAVE =================

result.to_csv(OUT_FILE, index=False)

print("\n================ SUMMARY ================\n")

print(f"Total stars : {len(result)}")
print(f"Prediction 1: {(result['Prediction'] == 1).sum()}")
print(f"Prediction 0: {(result['Prediction'] == 0).sum()}")

print(f"\nSaved: {OUT_FILE}")
