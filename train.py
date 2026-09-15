
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

FILE = r"astrobit_detections.csv"
TRUTH_FILE = r"train_truth.csv"
MODEL_FILE = r"astrobit_rf.pkl"
OUT_FILE = r"astrobit_confidence.csv"

FEATURES = [
    "Period_days",
    "Depth_ppm",
    "Duration_hours",
    "SDE"
]

df = pd.read_csv(FILE)
truth = pd.read_csv(TRUTH_FILE)

print("\nColumns found:")
print(df.columns.tolist())

df["KIC"] = df["KIC"].astype(str)
truth["kepid"] = truth["kepid"].astype(str)

# ================= INJECTED =================

injected_ids = set(
    truth.loc[truth["injected"] == 1, "kepid"]
)

df["injected"] = df["KIC"].isin(injected_ids).astype(int)

# ================= FINAL TARGET =================
# actual = label OR injected

df["actual"] = (
    (df["label"].astype(int) == 1) |
    (df["injected"] == 1)
).astype(int)

# ================= CLEAN =================

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + ["actual"]).copy()

X = df[FEATURES]
y = df["actual"].astype(int)

print("\n================ TRAINING ================\n")

print(f"Samples : {len(df)}")
print(f"0       : {(y == 0).sum()}")
print(f"1       : {(y == 1).sum()}")
print(f"Inputs  : {FEATURES}")

print("\nTarget breakdown:")
print(pd.crosstab(df["label"], df["injected"], margins=True))

# ================= MODEL =================

model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=8,
    min_samples_leaf=2,
    min_samples_split=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# ================= TRAINING OUTPUT =================

df["Confidence"] = model.predict_proba(X)[:, 1]
df["Prediction"] = (df["Confidence"] >= 0.5).astype(int)

df["Rp_Rs_pred"] = np.sqrt(
    np.maximum(df["Depth_ppm"], 0) / 1_000_000
)

df = df.sort_values(
    "Confidence",
    ascending=False
).reset_index(drop=True)

df.to_csv(OUT_FILE, index=False)

joblib.dump(model, MODEL_FILE)

# ================= TOP 30 =================

print("\n================ TOP 30 ================\n")

print(
    df[
        [
            "KIC",
            "Prediction",
            "Confidence",
            "actual",
            "label",
            "injected",
            "Period_days",
            "Depth_ppm",
            "Duration_hours",
            "SDE",
            "Rp_Rs_pred"
        ]
    ].head(30).to_string(index=False)
)

print("\n================ FEATURE IMPORTANCE ================\n")

for feature, importance in zip(FEATURES, model.feature_importances_):
    print(f"{feature:20s}: {importance:.4f}")

print("\n================ DONE ================\n")

print(f"Model saved  : {MODEL_FILE}")
print(f"Output saved : {OUT_FILE}")
