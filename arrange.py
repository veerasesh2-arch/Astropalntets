import pandas as pd

LABELS_FILE=r"C:\Users\LSESH\OneDrive\Desktop\prjs\Astrobit\train_labels.csv"
DETECT_FILE="astrobit_detections.csv"
OUT_FILE="train_labels_reordered.csv"

labels=pd.read_csv(LABELS_FILE)
det=pd.read_csv(DETECT_FILE)

labels["kepid"]=labels["kepid"].astype(str)
det["KIC"]=det["KIC"].astype(str)

order=pd.DataFrame({"kepid":det["KIC"]})
out=order.merge(labels,on="kepid",how="left")

out.to_csv(OUT_FILE,index=False)

print(f"Saved: {OUT_FILE}")
print(f"Detection rows : {len(det)}")
print(f"Output rows    : {len(out)}")
print(out.head(10).to_string(index=False))