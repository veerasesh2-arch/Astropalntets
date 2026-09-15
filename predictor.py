import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "astrobit_intial.csv")
MODEL_FILE = os.path.join(BASE_DIR, "astrobit_rf.pkl")
OUT_FILE = os.path.join(BASE_DIR, "astrobit_secondary.csv")

FEATURES = ["Period_days", "Depth_ppm", "Duration_hours", "SDE"]

def main():
    df = pd.read_csv(INPUT_FILE)
    model = joblib.load(MODEL_FILE)

    X = df[FEATURES]
    probabilities = model.predict_proba(X)[:, 1]

    df["Confidence_%"] = probabilities * 100

    df = df.sort_values("Confidence_%", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_FILE, index=False)

    print("\n================ PREDICTIONS ================\n")
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_FILE}")

if __name__ == "__main__":
    main()
