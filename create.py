import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

FILE="astrobit_detections.csv"
MODEL_FILE="astrobit_rf.pkl"

features=["Period_days","Depth_ppm","Duration_hours","SDE","kepmag","teff","logg","radius"]

df=pd.read_csv(FILE)
df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=features+["label"]).copy()

X=df[features]
y=df["label"].astype(int)

print("\n================ TRAINING ================\n")
print(f"Samples  : {len(df)}")
print(f"Features : {len(features)}")
print(f"Negative : {(y==0).sum()}")
print(f"Positive : {(y==1).sum()}")

model=RandomForestClassifier(
    n_estimators=1000,
    max_depth=8,
    min_samples_leaf=2,
    min_samples_split=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X,y)

joblib.dump(model,MODEL_FILE)

print("\n================ DONE ================\n")
print(f"Model saved: {MODEL_FILE}")