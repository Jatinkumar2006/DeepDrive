"""
DeepDrive Project: Step 0 - Data Preparation
=============================================
Module 1 of the DeepDrive pipeline. 
This script reads the raw smartphone accelerometer and gyroscope data from the local dataset folder,
applies robust data cleaning, and extracts overlapping sequential windows suitable for 
deep learning architectures. The output of this module feeds directly into step1_train.py.

Input: Raw sensor CSVs categorized by star ratings (1 to 5).
Output: Standardized sequence arrays (X.npy, y.npy) ready for training.
"""

# Data Preparation Pipeline
# This script is the foundation of the project. It aggregates raw sensor CSVs, standardizes labels,
# and applies a rigorous 11-step cleaning process to ensure high-quality training sequences.
import os
import glob
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

# Dataset path — local dataset/ folder inside this project
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'dataset'
)

RATING_MAP  = {'1 star': 1, '2 star': 2, '3 star': 3, '4 star': 4, '5 star': 5}
SENSOR_COLS = ['X_Acc', 'Y_Acc', 'Z_Acc', 'X_Gyro', 'Y_Gyro', 'Z_Gyro']
ACC_LIMIT   = 40.0
GYRO_LIMIT  = 300.0
GAP_MS      = 5000

# Window config
WINDOW = 100
STEP   = 50


# STEP A — RESTRUCTURE
def restructure() -> str:
    """
    Reads star folders, drops ID column, adds Rating, merges all CSVs.
    Returns path to the merged CSV.
    """
    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
    output_path = os.path.join('data', 'raw', 'all_sessions.csv')

    dataset_path = os.path.normpath(DATASET_PATH)
    if not os.path.exists(dataset_path):
        print(f"  ERROR: Dataset folder not found at:\n    {dataset_path}")
        print("  Edit DATASET_PATH in step0_prepare.py to point to your dataset.")
        raise SystemExit(1)

    all_frames  = []
    global_srno = 1

    for folder_name, rating in RATING_MAP.items():
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"  [SKIP] '{folder_name}' not found — rating {rating} skipped")
            continue

        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        print(f"\n  [{folder_name}]  Rating={rating}  |  {len(csv_files)} files")

        for fpath in sorted(csv_files):
            fname = os.path.basename(fpath)
            try:
                df = pd.read_csv(fpath, low_memory=False)

                # Drop ID / Name column
                drop_cols = [c for c in df.columns if c.strip().upper() in ('ID', 'NAME')]
                df = df.drop(columns=drop_cols, errors='ignore')

                # Check required sensor columns
                missing = [c for c in ['Timestamp'] + SENSOR_COLS if c not in df.columns]
                if missing:
                    print(f"    [SKIP] {fname} — missing: {missing}")
                    continue

                # Rating from folder name
                df['Rating'] = rating

                # Globally unique SrNo
                df['SrNo'] = range(global_srno, global_srno + len(df))
                global_srno += len(df)

                all_frames.append(df)
                print(f"    OK  {fname}  →  {len(df):,} rows")

            except Exception as e:
                print(f"    [ERROR] {fname}: {e}")

    if not all_frames:
        print("\n  ERROR: No data loaded. Check your folder structure.")
        raise SystemExit(1)

    merged = pd.concat(all_frames, ignore_index=True)
    merged['SrNo'] = range(1, len(merged) + 1)

    keep   = ['SrNo', 'Timestamp', 'X_Acc', 'Y_Acc', 'Z_Acc',
              'X_Gyro', 'Y_Gyro', 'Z_Gyro', 'Rating']
    merged = merged[[c for c in keep if c in merged.columns]]
    merged.to_csv(output_path, index=False)

    print(f"\n  {'='*50}")
    print(f"  RESTRUCTURE COMPLETE")
    print(f"  Total rows : {len(merged):,}")
    print(f"  Output     : {output_path}")
    print(f"  Rating distribution:")
    for r, cnt in merged['Rating'].value_counts().sort_index().items():
        pct = 100 * cnt / len(merged)
        print(f"    Rating {r}: {cnt:>7,} rows ({pct:.1f}%)")
    print(f"  {'='*50}")

    return output_path


# STEP B — CLEAN  (11 cleaning steps)
def clean(filepath: str) -> pd.DataFrame:
    """
    Runs all 11 cleaning steps for clean, reliable sensor data.
    """
    df     = pd.read_csv(filepath, low_memory=False)
    report = {'original_rows': len(df)}

    # 1. Exact duplicate rows
    n = len(df); df = df.drop_duplicates()
    report['exact_dupes_removed'] = n - len(df)

    # 2. Duplicate SrNo
    if 'SrNo' in df.columns:
        n = len(df)
        df = df.drop_duplicates(subset=['SrNo'], keep='first').reset_index(drop=True)
        report['srno_dupes_removed'] = n - len(df)

    # 3. NaN — forward-fill, then drop remaining
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill()
    n = len(df)
    df = df.dropna(subset=SENSOR_COLS + ['Rating', 'Timestamp'])
    report['nan_rows_removed'] = n - len(df)

    # 4. All-zero sensor rows
    n = len(df)
    df = df[~(df[SENSOR_COLS] == 0).all(axis=1)].reset_index(drop=True)
    report['all_zero_removed'] = n - len(df)

    # 5. Z_Acc == 0 (Gravity makes a persistent 0 value on the Z-axis physically impossible on a real device)
    n = len(df)
    df = df[df['Z_Acc'] != 0].reset_index(drop=True)
    report['z_acc_zero_removed'] = n - len(df)

    # 6. All gyro axes zero simultaneously
    n = len(df)
    df = df[~((df['X_Gyro']==0) & (df['Y_Gyro']==0) & (df['Z_Gyro']==0))].reset_index(drop=True)
    report['all_gyro_zero_removed'] = n - len(df)

    # 7. Frozen/stuck sensor (rolling std = 0 over 10 rows)
    n = len(df)
    for col in SENSOR_COLS:
        rstd = df[col].rolling(window=10, min_periods=10).std()
        df   = df[~((rstd == 0) & rstd.notna())].reset_index(drop=True)
    report['frozen_rows_removed'] = n - len(df)

    # 8. Hard physical limits
    for col in ['X_Acc', 'Y_Acc', 'Z_Acc']:
        df[col] = df[col].clip(-ACC_LIMIT, ACC_LIMIT)
    for col in ['X_Gyro', 'Y_Gyro', 'Z_Gyro']:
        df[col] = df[col].clip(-GYRO_LIMIT, GYRO_LIMIT)

    # 9. Statistical clip (1st–99th percentile per axis)
    for col in SENSOR_COLS:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    # 10. Sort by timestamp, assign session_id
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    df['session_id'] = (df['Timestamp'].diff().fillna(0) > GAP_MS).cumsum()
    report['session_segments'] = int(df['session_id'].nunique())

    # 11. Rating validation
    n = len(df)
    df = df[df['Rating'].between(1, 5)].reset_index(drop=True)
    df['Rating'] = df['Rating'].astype(int)
    report['invalid_rating_removed'] = n - len(df)

    df['SrNo'] = range(1, len(df) + 1)

    report['final_rows']    = len(df)
    report['rows_removed']  = report['original_rows'] - len(df)
    report['retention_pct'] = round(100 * len(df) / report['original_rows'], 2) # type: ignore

    print('\n  ======= CLEANING AUDIT REPORT =======')
    for k, v in report.items():
        print(f'    {k:<32}: {v}')
    print('  ======================================\n')

    return df


# STEP C — BUILD SEQUENCES  (DL-specific windowing)
def _safe_mode(series):
    m = series.mode()
    return m.iloc[0] if len(m) > 0 else series.iloc[0]


def build_sequences(df: pd.DataFrame):
    """Slide a 100-row window over each session segment, step 50."""
    X_list, y_list = [], []

    for _, sdf in df.groupby('session_id'):
        sdf  = sdf.reset_index(drop=True)
        vals = sdf[SENSOR_COLS].values.astype(np.float32)
        rats = sdf['Rating'].astype(int).values

        for i in range(0, len(sdf) - WINDOW + 1, STEP):
            X_list.append(vals[i: i + WINDOW])
            y_list.append(int(_safe_mode(pd.Series(rats[i: i + WINDOW]))))

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int32))


# MAIN
def main():
    print('\n' + '='*55)
    print('  Step 0 — Data preparation')
    print('='*55)
    print(f'  Dataset path: {os.path.normpath(DATASET_PATH)}')

    # A — Restructure
    print('\n  [A] Restructuring star folders …')
    raw_path = restructure()

    # B — Clean
    print('\n  [B] Cleaning …')
    os.makedirs(os.path.join('data', 'clean'), exist_ok=True)
    clean_df = clean(raw_path)
    clean_path = os.path.join('data', 'clean', 'all_sessions_clean.csv')
    clean_df.to_csv(clean_path, index=False)
    print(f'  Clean data saved → {clean_path}  ({len(clean_df):,} rows)')

    # C — Build sequences
    print('\n  [C] Building sequences …')
    X, y = build_sequences(clean_df)

    print(f'  Windows : {X.shape[0]:,}')
    print(f'  Shape   : {X.shape}  (windows × timesteps × sensors)')
    print(f'\n  Window label distribution:')
    for r, cnt in zip(*np.unique(y, return_counts=True)):
        pct = 100 * cnt / len(y)
        print(f'    Rating {r}: {cnt:>6,} ({pct:.1f}%)')

    # Normalise (per-feature z-score over entire training set)
    mu  = X.mean(axis=(0, 1), keepdims=True)
    sig = X.std(axis=(0, 1),  keepdims=True) + 1e-8
    X_n = (X - mu) / sig

    # Stratified train / test split (80/20)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X_n))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    # Save
    os.makedirs('data', exist_ok=True)
    np.save('data/sequences_X.npy',    X_n)
    np.save('data/sequences_y.npy',    y)
    joblib.dump({'mean': mu, 'std': sig},          'data/dl_scaler.pkl')
    joblib.dump({'train': tr_idx, 'test': te_idx}, 'data/split_indices.pkl')

    print(f'\n  Train windows : {len(tr_idx):,}')
    print(f'  Test  windows : {len(te_idx):,}')
    print(f'\n  Saved → data/sequences_X.npy')
    print(f'  Saved → data/sequences_y.npy')
    print(f'  Saved → data/dl_scaler.pkl')
    print(f'  Saved → data/split_indices.pkl')
    print(f'\n  Next: python step1_train.py')


if __name__ == '__main__':
    main()
