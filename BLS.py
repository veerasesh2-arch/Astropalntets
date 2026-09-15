import os
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "test")
OUT_FILE = os.path.join(BASE_DIR, "astrobit_intial.csv")

WORKERS = 12
P_MIN, P_MAX = 30, 290
N_COARSE, N_FINE, N_PEAKS = 40000, 800, 20
DURATIONS = np.arange(5, 15.01, 0.5) / 24

def clean(df):
    m = (df.quality.values == 0) & np.isfinite(df.flux.values)
    t = df.time.values[m]
    f = df.flux.values[m].astype(float)
    q = df.quarter.values[m]
    if len(t) < 1000:
        return None, None
    for qq in np.unique(q):
        s = q == qq
        med = np.median(f[s])
        f[s] = f[s] / med if med > 0 else 1
    cadence = np.median(np.diff(t))
    k = max(5, int(1.0 / cadence) | 1)
    trend = pd.Series(f).rolling(k, center=True, min_periods=k // 3).median().values
    ok = np.isfinite(trend) & (trend > 0)
    return t[ok], f[ok] / trend[ok]

def get_sde(power, i):
    med = np.nanmedian(power)
    mad = np.nanmedian(np.abs(power - med))
    return float((power[i] - med) / (1.4826 * mad)) if mad > 0 else 0.0

def get_candidates(t, f):
    bls = BoxLeastSquares(t, f)
    pmax = min(P_MAX, (t.max() - t.min()) / 3)
    if pmax <= P_MIN:
        return []

    coarse = np.exp(np.linspace(np.log(P_MIN), np.log(pmax), N_COARSE))
    r = bls.power(coarse, DURATIONS, objective="likelihood")
    power = np.asarray(r.power)
    order = np.argsort(power)[::-1]

    peaks, used = [], np.zeros(len(coarse), bool)
    for i in order:
        if used[i]:
            continue
        peaks.append(i)
        lo = np.searchsorted(coarse, coarse[i] * 0.9)
        hi = np.searchsorted(coarse, coarse[i] * 1.1)
        used[lo:hi] = True
        if len(peaks) >= N_PEAKS:
            break

    out = []
    for i in peaks:
        p0 = coarse[i]
        fine = np.linspace(max(P_MIN, p0 * 0.98), min(pmax, p0 * 1.02), N_FINE)
        if len(fine) < 10:
            continue
        rr = bls.power(fine, DURATIONS, objective="likelihood")
        j = int(np.nanargmax(rr.power))
        out.append({
            "period": float(rr.period[j]),
            "depth_ppm": float(rr.depth[j] * 1e6),
            "duration_h": float(rr.duration[j] * 24),
            "epoch": float(rr.transit_time[j]),
            "sde": get_sde(power, i)
        })
    return sorted(out, key=lambda x: x["sde"], reverse=True)

def harmonic(p1, p2):
    for r in [0.5, 1 / 3, 2, 3]:
        if abs(p2 - p1 * r) / (p1 * r) <= 0.02:
            return True
    return False

def choose(candidates):
    if not candidates:
        return None
    top = candidates[0]
    family = [top]
    for c in candidates[1:]:
        if harmonic(top["period"], c["period"]):
            family.append(c)
    return min(family, key=lambda x: x["period"])

def process_star(file):
    path = os.path.join(DATA_DIR, file)
    kepid = os.path.splitext(file)[0].replace("KIC_", "")
    try:
        df = pd.read_parquet(path)
        t, f = clean(df)
        if t is None:
            return kepid, None, "CLEAN FAILED"
        best = choose(get_candidates(t, f))
        if best is None:
            return kepid, None, "NO CANDIDATE"
        row = {
            "KIC": kepid,
            "Period_days": best["period"],
            "Depth_ppm": best["depth_ppm"],
            "Duration_hours": best["duration_h"],
            "Epoch": best["epoch"],
            "SDE": best["sde"]
        }
        return kepid, row, None
    except Exception as e:
        return kepid, None, f"ERROR {e}"

def main():
    files = sorted(x for x in os.listdir(DATA_DIR) if x.lower().endswith(".parquet"))
    if not files:
        print(f"No Parquet files found in: {DATA_DIR}")
        return

    results = []
    print(f"Found {len(files)} Parquet files.")
    print(f"Processing with {WORKERS} workers...\n")

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_star, f): f for f in files}
        for done, future in enumerate(as_completed(futures), 1):
            kepid, row, error = future.result()
            if row:
                results.append(row)
                print(
                    f"{done}/{len(files)} KIC_{kepid}: "
                    f"P={row['Period_days']:.3f} d | "
                    f"depth={row['Depth_ppm']:.1f} ppm | "
                    f"dur={row['Duration_hours']:.1f} h | "
                    f"SDE={row['SDE']:.2f}"
                )
            else:
                print(f"{done}/{len(files)} KIC_{kepid}: {error}")

    out = pd.DataFrame(results)
    if len(out):
        out = out.sort_values("SDE", ascending=False).reset_index(drop=True)
        out.to_csv(OUT_FILE, index=False)

    print("\n================ FINAL TABLE ================\n")
    print(out.to_string(index=False))
    print(f"\nSaved: {OUT_FILE}")

if __name__ == "__main__":
    main()
