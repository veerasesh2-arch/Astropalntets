import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def clean_and_flatten_lightcurve(file_path, output_dir):
    """
    Cleans, normalizes across quarters, and flattens stellar trend arches 
    without distorting transit depths or scrambling time order.
    """
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        return

    # Filter bad quality data points if column exists
    if 'quality' in df.columns:
        df = df[df['quality'] == 0].copy()
    
    # Sort strictly by time before processing
    df = df.sort_values('time').reset_index(drop=True)
    
    # Normalize per quarter to unity baseline
    if 'quarter' in df.columns:
        df['flux_norm'] = df.groupby('quarter')['flux'].transform(lambda x: x / (x.median() + 1e-8))
    else:
        df['quarter'] = 1
        df['flux_norm'] = df['flux'] / (df['flux'].median() + 1e-8)

    # Pre-allocate array to preserve row index matching
    df['flux_flat'] = np.nan

    # Process each quarter individually safely
    for quarter, group in df.groupby('quarter'):
        group_idx = group.index
        fluxes = group['flux_norm'].values
        n_points = len(fluxes)

        # Dynamic window size setup (~20-day window assuming 30-min cadence)
        window_size = 1001
        poly_order = 2 

        # Guarantee odd window size smaller than array length
        if n_points <= window_size:
            window_size = n_points if n_points % 2 != 0 else n_points - 1

        if window_size > poly_order and window_size >= 7:
            # Pass 1: Initial trend line
            trend_pass1 = savgol_filter(fluxes, window_length=window_size, polyorder=poly_order)
            residual = fluxes - trend_pass1
            
            # Mask out transit dips (> 2.5 sigma below baseline)
            std_dev = np.std(residual)
            mask = residual > (-2.5 * std_dev)
            
            # Interpolate masked transit points with pass 1 trend line
            fluxes_cleaned = np.where(mask, fluxes, trend_pass1)
            
            # Pass 2: Clean trend fit excluding transit points
            trend_final = savgol_filter(fluxes_cleaned, window_length=window_size, polyorder=poly_order)
            flat_flux = fluxes / (trend_final + 1e-8)
        else:
            flat_flux = fluxes

        # Assign back using exact dataframe indices to prevent order mismatch
        df.loc[group_idx, 'flux_flat'] = flat_flux

    # Save Output
    base_name = os.path.basename(file_path)
    file_name = base_name.replace('.parquet', '_flat.csv').replace('.csv', '_flat.csv')
    output_path = os.path.join(output_dir, file_name)
    
    df[['time', 'flux_flat', 'quarter']].to_csv(output_path, index=False)
    print(f"✅ Successfully flattened: {file_name}")


def process_folder(input_folder, output_dir="flattened_csvs"):
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.csv', '.parquet')
    files = [f for f in os.listdir(input_folder) if f.endswith(valid_extensions)]
    
    print(f"Found {len(files)} files to process in: {input_folder}\n")
    
    for file in files:
        full_path = os.path.join(input_folder, file)
        try:
            clean_and_flatten_lightcurve(full_path, output_dir)
        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")

if __name__ == "__main__":
    input_folder_path = r"C:\Users\LSESH\OneDrive\Desktop\prjs\Astrobit\dev"
    process_folder(input_folder_path, output_dir="flattenedev_csvs")