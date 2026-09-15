import pandas as pd
import numpy as np

DET_FILE="astrobit_dev.csv"
TRUTH_FILE="dev_truth.csv"
OUT_FILE="astrobit_dev_check.csv"
TOLERANCE=0.3

det=pd.read_csv(DET_FILE)
truth=pd.read_csv(TRUTH_FILE)

det["KIC"]=det["KIC"].astype(str)
truth["kepid"]=truth["kepid"].astype(str)

df=det.merge(truth,left_on="KIC",right_on="kepid",how="inner")

def check_period(found,true):
    aliases=[true,true*2,true/2,true*3]
    errors=[abs(found-a)/a for a in aliases]
    i=int(np.argmin(errors))
    return errors[i]<=TOLERANCE,aliases[i],errors[i]*100

results=[]
for _,r in df.iterrows():
    found=r["Period_days"]
    true=r["period_days"]
    if pd.isna(found) or pd.isna(true):
        continue
    hit,matched,error=check_period(found,true)
    results.append({
        "KIC":r["KIC"],
        "Injected":r["injected"],
        "True_Period_days":true,
        "Detected_Period_days":found,
        "Matched_Period_days":matched,
        "Period_Error_%":error,
        "Period_HIT":hit,
        "True_Depth_ppm":r["depth_ppm"],
        "Detected_Depth_ppm":r["Depth_ppm"],
        "True_Duration_h":r["duration_hours"],
        "Detected_Duration_h":r["Duration_hours"],
        "SDE":r["SDE"]
    })

out=pd.DataFrame(results)
out.to_csv(OUT_FILE,index=False)

print("\n================ DEV VERIFICATION ================\n")
print(out.head(30).to_string(index=False))

injected=out[out["Injected"]==1]

print("\n================ SUMMARY ================\n")
print(f"Compared stars : {len(out)}")
print(f"Injected stars : {len(injected)}")
print(f"Period HITs    : {injected['Period_HIT'].sum()}")
print(f"Recall         : {injected['Period_HIT'].mean()*100:.2f}%")
print(f"Tolerance      : {TOLERANCE*100:.0f}%")

print(f"\nSaved: {OUT_FILE}")