
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BLS_FILE = r"C:\Users\LSESH\OneDrive\Desktop\prjs\Astrobit\astrobit_dev.csv"
LABELS_FILE = r"C:\Users\LSESH\OneDrive\Desktop\prjs\Astrobit\dev_labels.csv"
TRUTH_FILE = r"C:\Users\LSESH\OneDrive\Desktop\prjs\Astrobit\dev_truth.csv"
MODEL_FILE = r"C:\Users\LSESH\OneDrive\Desktop\prjs\Astrobit\astrobit_rf.pkl"
OUTPUT_FILE = r"C:\Users\LSESH\OneDrive\Desktop\prjs\Astrobit\astrobit_dev_predictions.csv"

FEATURES = ["Period_days", "Depth_ppm", "Duration_hours", "SDE"]

# ================= LOAD =================

bls = pd.read_csv(BLS_FILE)
labels = pd.read_csv(LABELS_FILE)
truth = pd.read_csv(TRUTH_FILE)

bls["KIC"] = bls["KIC"].astype(str)
labels["kepid"] = labels["kepid"].astype(str)
truth["kepid"] = truth["kepid"].astype(str)

# ================= INJECTED =================
# dev_truth contains the injected stars.
# If KIC exists in dev_truth -> injected = 1
# Otherwise -> injected = 0

injected_ids = set(truth["kepid"])
bls["injected"] = bls["KIC"].isin(injected_ids).astype(int)

# True injected period, only available for injected stars
truth_period = truth.set_index("kepid")["period_days"]
bls["true_period"] = bls["KIC"].map(truth_period)

# ================= DEV LABEL =================

label_map = labels.set_index("kepid")["label"]
bls["label"] = bls["KIC"].map(label_map).fillna(0).astype(int)

# ================= FINAL ACTUAL =================
# Actual = 1 if either label OR injection is 1

bls["actual"] = ((bls["label"] == 1) | (bls["injected"] == 1)).astype(int)

# ================= LOAD MODEL =================

model = joblib.load(MODEL_FILE)

X = bls[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

prediction = model.predict(X)
confidence = model.predict_proba(X)[:, 1]

bls["Prediction"] = prediction.astype(int)
bls["Confidence_%"] = confidence * 100

# ================= TOTAL DEV =================

y_true = bls["actual"]
y_pred = bls["Prediction"]

print("\n================ TOTAL DEV SET ================\n")

print(f"Total DEV stars    : {len(bls)}")
print(f"Actual class 1     : {y_true.sum()}")
print(f"Actual class 0     : {(y_true == 0).sum()}")
print(f"Predicted 1        : {(y_pred == 1).sum()}")
print(f"Predicted 0        : {(y_pred == 0).sum()}")

# ================= ML RESULTS =================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, confidence)

print("\n================ ACCURACY ================\n")

print(f"Correct predictions : {(y_true == y_pred).sum()}/{len(bls)}")
print(f"Wrong predictions   : {(y_true != y_pred).sum()}/{len(bls)}")
print(f"Accuracy            : {accuracy * 100:.2f}%")

print("\n================ ML RESULTS ================\n")

print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1        : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ================= ALL PREDICTIONS =================

result = bls[
    [
        "KIC",
        "Prediction",
        "Confidence_%",
        "actual",
        "label",
        "injected",
        "Period_days",
        "true_period"
    ]
].copy()

result = result.sort_values("Confidence_%", ascending=False)

print("\n================ ALL 89 PREDICTIONS ================\n")
print(result.to_string(index=False))

# ================= INJECTED PERIOD CHECK =================

inj = bls[bls["injected"] == 1].copy()

def period_hit(row):
    p = row["Period_days"]
    true_p = row["true_period"]

    if not np.isfinite(p) or not np.isfinite(true_p):
        return 0

    aliases = [
        true_p,
        true_p / 2,
        true_p * 2,
        true_p / 3,
        true_p * 3
    ]

    return int(
        any(abs(p - a) / a <= 0.02 for a in aliases)
    )

def period_error(row):
    p = row["Period_days"]
    true_p = row["true_period"]

    if not np.isfinite(p) or not np.isfinite(true_p):
        return np.nan

    return abs(p - true_p) / true_p * 100

inj["Period_Error_%"] = inj.apply(period_error, axis=1)
inj["Period_HIT_2%"] = inj.apply(period_hit, axis=1)

period_hits = inj["Period_HIT_2%"].sum()
period_total = len(inj)

print("\n================ INJECTED PERIOD CHECK ================\n")

print(
    inj[
        [
            "KIC",
            "Prediction",
            "Confidence_%",
            "Period_days",
            "true_period",
            "Period_Error_%",
            "Period_HIT_2%"
        ]
    ].sort_values("Confidence_%", ascending=False).to_string(index=False)
)

print(f"\nBLS period hits within 2% : {period_hits}/{period_total}")
print(f"BLS period recall         : {period_hits / period_total * 100:.2f}%")

# ================= SAVE =================

result.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE}")
