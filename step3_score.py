"""
DeepDrive Project: Step 3 - Real-Time Inference
================================================
Module 4 of the DeepDrive pipeline.
This script acts as the command-line inference engine. It applies the unified 11-step cleaning 
pipeline (from Step 0) to a brand new, unseen sensor CSV, loads the best trained model (from Step 1), 
and outputs a continuous 0-100 driving safety score in real-time.

Usage:
  python step3_score.py path/to/session.csv [--model lstm|gru|transformer]
"""

# Inference Pipeline
# This script loads the best performing trained model and uses it to score completely new, unseen raw sensor sessions in real-time.
import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

WINDOW      = 100
STEP        = 50
SENSOR_COLS = ['X_Acc', 'Y_Acc', 'Z_Acc', 'X_Gyro', 'Y_Gyro', 'Z_Gyro']
ACC_LIMIT   = 40.0
GYRO_LIMIT  = 300.0
GAP_MS      = 5000

# 100 = safest (Rating 5), 0 = most dangerous (Rating 1)
RISK_W = np.array([0, 25, 50, 75, 100], dtype=np.float32)


def score_to_label(score):
    """Convert 0-100 driving score to band, color, and stars."""
    if score >= 80: return 'EXCELLENT',  '#22c55e', '★★★★★', 5.0
    if score >= 60: return 'GOOD',       '#84cc16', '★★★★☆', 4.0
    if score >= 40: return 'MODERATE',   '#f59e0b', '★★★☆☆', 3.0
    if score >= 20: return 'POOR',       '#ef4444', '★★☆☆☆', 2.0
    return                 'DANGEROUS',  '#991b1b', '★☆☆☆☆', 1.0


def score_to_stars(score):
    return round(score / 20, 1)


# Version-safe TransformerBlock
def _make_transformer_block(tf):
    keras = tf.keras
    register = None
    for _try in [
        lambda: keras.saving.register_keras_serializable,
        lambda: keras.utils.register_keras_serializable,
        lambda: __import__('keras').saving.register_keras_serializable,
    ]:
        try:
            fn = _try()
            if callable(fn):
                register = fn
                break
        except Exception:
            pass

    class TransformerBlock(keras.layers.Layer):
        def __init__(self, d, heads, ff, drop=0.2, **kw):
            super().__init__(**kw)
            self.d = d; self.heads = heads
            self.ff = ff; self.drop = drop
            self.attn  = keras.layers.MultiHeadAttention(
                num_heads=heads, key_dim=d // heads)
            self.ffn   = keras.Sequential([
                keras.layers.Dense(ff, activation='gelu'),
                keras.layers.Dense(d),
            ])
            self.ln1   = keras.layers.LayerNormalization(epsilon=1e-6)
            self.ln2   = keras.layers.LayerNormalization(epsilon=1e-6)
            self.drop1 = keras.layers.Dropout(drop)
            self.drop2 = keras.layers.Dropout(drop)

        def call(self, x, training=False):
            x = self.ln1(x + self.drop1(self.attn(x, x), training=training))
            x = self.ln2(x + self.drop2(self.ffn(x),     training=training))
            return x

        def get_config(self):
            cfg = super().get_config()
            cfg.update(d=self.d, heads=self.heads, ff=self.ff, drop=self.drop)
            return cfg

    if callable(register):
        try:
            TransformerBlock = register( # type: ignore
                package='DrivingRisk', name='TransformerBlock')(TransformerBlock)
        except Exception:
            pass
    return TransformerBlock


def clean_session(df: pd.DataFrame) -> pd.DataFrame:
    # Re-use the exact same 11-step cleaning logic from the training pipeline to ensure data consistency during inference
    drop = [c for c in df.columns if c.strip().upper() in ('ID', 'NAME')]
    df   = df.drop(columns=drop, errors='ignore')
    if 'Rating' not in df.columns:
        df['Rating'] = 3
    df[SENSOR_COLS] = df[SENSOR_COLS].apply(pd.to_numeric, errors='coerce')
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill().bfill()
    df = df.dropna(subset=SENSOR_COLS)
    df = df[~(df[SENSOR_COLS] == 0).all(axis=1)] # type: ignore
    df = df[df['Z_Acc'] != 0]
    for col in ['X_Acc', 'Y_Acc', 'Z_Acc']:
        df[col] = df[col].clip(-ACC_LIMIT, ACC_LIMIT)
    for col in ['X_Gyro', 'Y_Gyro', 'Z_Gyro']:
        df[col] = df[col].clip(-GYRO_LIMIT, GYRO_LIMIT)
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    df['session_id'] = (df['Timestamp'].diff().fillna(0) > GAP_MS).cumsum()
    return df.reset_index(drop=True)


def build_windows(df: pd.DataFrame) -> np.ndarray:
    """Apply the SAME normalisation used during training (loaded from scaler pkl)."""
    scaler  = joblib.load('data/dl_scaler.pkl')
    mu, sig = scaler['mean'], scaler['std']
    windows = []
    for _, sdf in df.groupby('session_id'):
        sdf  = sdf.reset_index(drop=True)
        vals = sdf[SENSOR_COLS].values.astype(np.float32)
        for i in range(0, len(sdf) - WINDOW + 1, STEP):
            windows.append(vals[i: i + WINDOW])
    if not windows:
        return np.empty((0, WINDOW, 6), dtype=np.float32)
    X = np.array(windows, dtype=np.float32)
    return (X - mu) / sig


def score_session(csv_path: str, model_name: str):
    print(f'\n  Scoring: {csv_path}')
    print(f'  Model  : {model_name.upper()}')

    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        print('  ERROR: pip install tensorflow')
        raise SystemExit(1)

    model_path = f'models/{model_name}_model.h5'
    if not os.path.exists(model_path):
        print(f'  ERROR: {model_path} not found. Run step1_train.py first.')
        raise SystemExit(1)

    if not os.path.exists('data/dl_scaler.pkl'):
        print('  ERROR: data/dl_scaler.pkl not found. Run step0_prepare.py first.')
        raise SystemExit(1)

    TransformerBlock = _make_transformer_block(tf)
    raw_df   = pd.read_csv(csv_path, low_memory=False)
    clean_df = clean_session(raw_df)
    print(f'  Rows after cleaning: {len(clean_df):,}')

    if len(clean_df) < WINDOW:
        print(f'  ERROR: session too short (need >= {WINDOW} rows, got {len(clean_df)})')
        raise SystemExit(1)

    X = build_windows(clean_df)
    if len(X) == 0:
        print('  ERROR: no complete windows extracted')
        raise SystemExit(1)
    print(f'  Windows extracted: {len(X)}')

    model  = tf.keras.models.load_model(
        model_path, custom_objects={'TransformerBlock': TransformerBlock})
    proba  = model.predict(X, verbose=0)
    scores = (proba * RISK_W).sum(axis=1)

    session_score         = float(scores.mean())
    band, color, stars, _ = score_to_label(session_score)
    star_num              = score_to_stars(session_score)

    excellent_n = int((scores >= 80).sum())
    good_n      = int(((scores >= 60) & (scores < 80)).sum())
    moderate_n  = int(((scores >= 40) & (scores < 60)).sum())
    poor_n      = int(((scores >= 20) & (scores < 40)).sum())
    dangerous_n = int((scores < 20).sum())

    print(f'\n  Driving score: {session_score:.1f} / 100  ({star_num}/5.0 {stars})  [{band}]')
    print(f'  Based on {len(scores)} windows:')
    print(f'    Excellent (80-100): {excellent_n}')
    print(f'    Good      (60-79) : {good_n}')
    print(f'    Moderate  (40-59) : {moderate_n}')
    print(f'    Poor      (20-39) : {poor_n}')
    print(f'    Dangerous (0-19)  : {dangerous_n}')


    # Plot score timeline
    os.makedirs('outputs', exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(range(len(scores)), scores, alpha=0.15, color=color)
    ax.plot(scores, lw=1.5, color=color)
    ax.axhline(80, color='#22c55e', ls='--', lw=0.8, alpha=0.6, label='Excellent (80)')
    ax.axhline(60, color='#84cc16', ls='--', lw=0.8, alpha=0.6, label='Good (60)')
    ax.axhline(40, color='#f59e0b', ls='--', lw=0.8, alpha=0.6, label='Moderate (40)')
    ax.axhline(20, color='#ef4444', ls='--', lw=0.8, alpha=0.6, label='Poor (20)')
    ax.set_xlabel('Window index (each approx 1-2 seconds)')
    ax.set_ylabel('Driving score (0=dangerous, 100=perfect)')
    ax.set_ylim(-2, 105)
    ax.set_title(f'{model_name.upper()} — Driving score: {session_score:.1f}/100  '
                 f'{stars}  {star_num}/5.0  [{band}]')
    ax.legend(fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('outputs/session_score.png', dpi=150)
    plt.close()
    print('\n  Timeline saved → outputs/session_score.png')

    return session_score, band


def main():
    parser = argparse.ArgumentParser(description='Score a driving session CSV')
    parser.add_argument('csv', nargs='?', help='Path to session CSV')
    parser.add_argument('--model', default=None,
                        choices=['lstm', 'gru', 'transformer'],
                        help='Model to use (default: best from evaluation)')
    args = parser.parse_args()

    if not args.csv:
        print('Usage: python step3_score.py path/to/session.csv [--model lstm|gru|transformer]')
        raise SystemExit(0)

    if not os.path.exists(args.csv):
        print(f'ERROR: File not found: {args.csv}')
        raise SystemExit(1)

    model_name = args.model
    if model_name is None:
        best_file = 'models/best_model.txt'
        if os.path.exists(best_file):
            with open(best_file) as f:
                model_name = f.read().strip()
            print(f'  Using best model: {model_name.upper()}')
        else:
            model_name = 'gru'
            print(f'  Defaulting to GRU')

    score_session(args.csv, model_name)


if __name__ == '__main__':
    main()
