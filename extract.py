import os
import pandas as pd

# 1. RELATIVE PATHS (Runs directly in the current folder)
INPUT_FOLDER = r"D:\Train-20260913T135023Z-1-001\Train\train_pack\train"  # Current directory where the script and parquet files live
OUTPUT_CSV_FOLDER = "./train"  # Will create a new folder named 'cleaned_csvs' here
SUMMARY_EXCEL = "./train.xlsx"  # Will create 'dataset_summary.xlsx' in this folder

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_CSV_FOLDER, exist_ok=True)

summary_list = []

# Get all parquet files in the current folder
files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".parquet")]
print(f"Found {len(files)} parquet files in current directory. Processing...\n")

for idx, filename in enumerate(files, 1):
    file_path = os.path.join(INPUT_FOLDER, filename)

    # Extract clean star ID (e.g., KIC_1719974.parquet -> 1719974)
    kepid = filename.replace("KIC_", "").replace(".parquet", "")

    # Read the parquet file
    df = pd.read_parquet(file_path)

    # Clean data (keep quality == 0)
    clean_df = df[df["quality"] == 0].copy()

    # Save cleaned light curve to CSV
    csv_filename = f"star_{kepid}_clean.csv"
    csv_path = os.path.join(OUTPUT_CSV_FOLDER, csv_filename)
    clean_df.to_csv(csv_path, index=False)

    # Extract summary metrics for Excel overview
    summary_list.append(
        {
            "Star ID (kepid)": kepid,
            "Original Parquet": filename,
            "Clean CSV Saved": csv_filename,
            "Total Cadences": len(df),
            "Clean Cadences": len(clean_df),
            "Start Time (Days)": round(clean_df["time"].min(), 2)
            if not clean_df.empty
            else None,
            "End Time (Days)": round(clean_df["time"].max(), 2)
            if not clean_df.empty
            else None,
            "Mean Flux": round(clean_df["flux"].mean(), 2)
            if not clean_df.empty
            else None,
            "Flux Std Dev (Noise)": round(clean_df["flux"].std(), 2)
            if not clean_df.empty
            else None,
        }
    )

    print(f"[{idx}/{len(files)}] Processed: {filename} -> Saved {csv_filename}")

# Create Excel Summary Table
summary_df = pd.DataFrame(summary_list)
summary_df.to_excel(SUMMARY_EXCEL, index=False)

print(
    f"\nSUCCESS! All CSVs saved to '{OUTPUT_CSV_FOLDER}' and summary saved to '{SUMMARY_EXCEL}'."
)