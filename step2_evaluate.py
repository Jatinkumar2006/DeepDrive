"""
DeepDrive Project: Step 2 - Model Evaluation
=============================================
Module 3 of the DeepDrive pipeline.
This script evaluates the trained models from step1_train.py against the held-out test set 
generated in step0_prepare.py. It automatically computes standard performance metrics and 
exports analytical charts to the outputs/ directory.

It determines the absolute best-performing architecture to serve as the default engine 
for the real-time inference module (step3_score.py) and the Streamlit dashboard (app.py).

Outputs: Analytical comparison plots (PNG) and benchmark results (CSV).
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.makedirs('outputs', exist_ok=True)

# 100 = safest (Rating 5), 0 = most dangerous (Rating 1)
RISK_W  = np.array([0, 25, 50, 75, 100], dtype=np.float32)
COLORS  = {'lstm': '#2563EB', 'gru': '#0E7490', 'transformer': '#854F0B'}
LABELS  = {'lstm': 'LSTM', 'gru': 'GRU', 'transformer': 'Transformer'}
RATINGS = [1, 2, 3, 4, 5]


def score_to_label(score):
    """Convert 0-100 driving score to band, color, and star rating."""
    if score >= 80:
        return 'Excellent',  '#22c55e', '★★★★★'
    if score >= 60:
        return 'Good',       '#84cc16', '★★★★☆'
    if score >= 40:
        return 'Moderate',   '#f59e0b', '★★★☆☆'
    if score >= 20:
        return 'Poor',       '#ef4444', '★★☆☆☆'
    return     'Dangerous',  '#991b1b', '★☆☆☆☆'


def score_to_stars(score):
    """Convert 0-100 score to x.x / 5.0 star rating."""
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


def load_model(tf, name):
    path = f'models/{name}_model.h5'
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found. Run step1_train.py first.')
    TransformerBlock = _make_transformer_block(tf)
    return tf.keras.models.load_model(
        path, custom_objects={'TransformerBlock': TransformerBlock})


# Evaluation
def evaluate_one(model, X_te, y_te):
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
    from sklearn.preprocessing import label_binarize

    proba = model.predict(X_te, verbose=0)
    pred0 = proba.argmax(axis=1)
    pred  = pred0 + 1  # back to 1-5

    f1  = f1_score(y_te, pred, average='weighted', zero_division=0)
    try:
        y_bin = label_binarize(y_te - 1, classes=[0, 1, 2, 3, 4])
        auc   = roc_auc_score(y_bin, proba, multi_class='ovr', average='weighted')
    except Exception:
        auc = float('nan')

    cm     = confusion_matrix(y_te, pred, labels=RATINGS)
    scores = (proba * RISK_W).sum(axis=1)  # 100=safe, 0=dangerous

    per_class = {}
    for r in RATINGS:
        mask = (y_te == r)
        if mask.sum() > 0:
            per_class[r] = round(float((pred[mask] == r).mean()), 4)

    return {
        'f1':        round(float(f1),  4),
        'auc':       round(float(auc), 4),
        'scores':    scores,
        'pred':      pred,
        'confusion': cm,
        'per_class': per_class,
    }


# Charts
def plot_training_curves(histories):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, name in enumerate(['lstm', 'gru', 'transformer']):
        h = histories[name]; c = COLORS[name]; lbl = LABELS[name]
        ax = axes[0, col]
        ax.plot(h['loss'],     color=c, lw=2,          label='Train loss')
        ax.plot(h['val_loss'], color=c, lw=2, ls='--', label='Val loss', alpha=0.65)
        ax.set_title(lbl, fontsize=13, fontweight='bold', color=c)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
        ax = axes[1, col]
        ax.plot(h['accuracy'],     color=c, lw=2,          label='Train acc')
        ax.plot(h['val_accuracy'], color=c, lw=2, ls='--', label='Val acc', alpha=0.65)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
    plt.suptitle('Training Curves — LSTM  ·  GRU  ·  Transformer',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/1_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/1_training_curves.png')


def plot_comparison(results, times):
    names = ['lstm', 'gru', 'transformer']
    lbls  = [LABELS[n] for n in names]
    f1s   = [results[n]['f1']  for n in names]
    aucs  = [results[n]['auc'] for n in names]
    ts    = [times[n]          for n in names]
    clrs  = [COLORS[n]         for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, vals, title, ylabel, fmt in zip(
        axes,
        [f1s, aucs, ts],
        ['Weighted F1\n(higher = better)',
         'AUC-ROC (OvR weighted)\n(higher = better)',
         'Training time (seconds)\n(lower = better)'],
        ['F1 score', 'AUC-ROC', 'Seconds'],
        ['%.4f', '%.4f', '%.0f'],
    ):
        bars = ax.bar(lbls, vals, color=clrs, edgecolor='white', linewidth=0.5, width=0.5)
        ax.bar_label(bars, fmt=fmt, padding=5, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
    plt.suptitle('Model Comparison — LSTM  vs  GRU  vs  Transformer',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/2_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/2_model_comparison.png')


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, ['lstm', 'gru', 'transformer']):
        r = results[name]
        # Normalize per row for clearer visualization
        cm_norm = r['confusion'].astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=[str(r) for r in RATINGS], yticklabels=[str(r) for r in RATINGS], ax=ax,
                    linewidths=0.3, linecolor='white', annot_kws={'size': 10},
                    vmin=0, vmax=1)
        ax.set_title(f'{LABELS[name]}\nF1 = {r["f1"]:.4f}', fontsize=12,
                     fontweight='bold', color=COLORS[name])
        ax.set_xlabel('Predicted rating', fontsize=10)
        ax.set_ylabel('True rating',      fontsize=10)
    plt.suptitle('Confusion Matrices (row-normalized) — Deep Learning Models',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/3_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/3_confusion_matrices.png')


def plot_score_distributions(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rc = {1:'#991b1b', 2:'#ef4444', 3:'#f59e0b', 4:'#84cc16', 5:'#22c55e'}
    for ax, name in zip(axes, ['lstm', 'gru', 'transformer']):
        r = results[name]
        for rating in RATINGS:
            mask = (r['pred'] == rating)
            if mask.sum() > 0:
                ax.hist(r['scores'][mask], bins=25, alpha=0.6,
                        color=rc[rating], label=f'Rating {rating}')
        ax.axvline(r['scores'].mean(), color='#1e293b', lw=1.5, ls='--',
                   label=f'Mean = {r["scores"].mean():.1f}')
        ax.set_title(LABELS[name], fontsize=12, fontweight='bold', color=COLORS[name])
        ax.set_xlabel('Driving score (0=dangerous, 100=perfect)', fontsize=10)
        ax.set_ylabel('Window count', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        if ax == axes[0]:
            ax.legend(fontsize=8, title='Predicted rating')
    plt.suptitle('Driving Score Distribution per Model\n'
                 'Rating 1 → score near 0 (dangerous)   ·   Rating 5 → score near 100 (perfect)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/4_risk_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/4_risk_distributions.png')


def plot_score_by_rating(results, y_te):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(RATINGS)); width = 0.25
    for i, name in enumerate(['lstm', 'gru', 'transformer']):
        r     = results[name]
        means = [r['scores'][y_te == rt].mean() if (y_te == rt).sum() > 0 else 0
                 for rt in RATINGS]
        bars = ax.bar(x + (i - 1) * width, means, width,
                      color=COLORS[name], label=LABELS[name],
                      edgecolor='white', linewidth=0.5)
        ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Rating {r}' for r in RATINGS], fontsize=10)
    ax.set_ylabel('Mean driving score (0–100)', fontsize=11)
    ax.set_title('Mean Driving Score per True Rating\n'
                 'Should increase from Rating 1 → Rating 5', fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('outputs/5_risk_by_rating.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/5_risk_by_rating.png')


def plot_per_class_accuracy(results):
    """Per-class accuracy bar chart — shows how well each rating is classified."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rc = {1:'#991b1b', 2:'#ef4444', 3:'#f59e0b', 4:'#84cc16', 5:'#22c55e'}
    for ax, name in zip(axes, ['lstm', 'gru', 'transformer']):
        r = results[name]
        ratings = list(r['per_class'].keys())
        accs    = [r['per_class'][rt] for rt in ratings]
        colors  = [rc[rt] for rt in ratings]
        bars = ax.bar([f'R{rt}' for rt in ratings], accs, color=colors,
                      edgecolor='white', linewidth=0.5)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Accuracy per class', fontsize=10)
        ax.set_title(f'{LABELS[name]}\nF1={r["f1"]:.4f}', fontsize=12,
                     fontweight='bold', color=COLORS[name])
        ax.spines[['top', 'right']].set_visible(False)
    plt.suptitle('Per-Class Accuracy — How well each rating is classified',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/6_per_class_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/6_per_class_accuracy.png')


# Main
def main():
    print('\n' + '='*55)
    print('  Step 2 — Evaluating models')
    print('='*55)

    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        print(f'  TensorFlow {tf.__version__}')
    except ImportError:
        print('  ERROR: TensorFlow not installed. pip install tensorflow')
        raise SystemExit(1)

    for f in ['data/sequences_X.npy', 'data/sequences_y.npy',
              'data/split_indices.pkl', 'models/train_history.pkl',
              'models/train_times.json']:
        if not os.path.exists(f):
            print(f'  ERROR: {f} missing.')
            print('  Run step0_prepare.py then step1_train.py first.')
            raise SystemExit(1)

    X      = np.load('data/sequences_X.npy')
    y      = np.load('data/sequences_y.npy')
    splits = joblib.load('data/split_indices.pkl')
    te_idx = splits['test']
    X_te   = X[te_idx]
    y_te   = y[te_idx]

    histories = joblib.load('models/train_history.pkl')
    with open('models/train_times.json') as f:
        times = json.load(f)

    print(f'\n  Test set : {len(X_te):,} windows')

    results = {}
    for name in ['lstm', 'gru', 'transformer']:
        print(f'\n  Loading {LABELS[name]} …')
        mdl = load_model(tf, name)
        res = evaluate_one(mdl, X_te, y_te)
        results[name] = res
        band, _, stars = score_to_label(res['scores'].mean())
        print(f'    F1           = {res["f1"]:.4f}')
        print(f'    AUC          = {res["auc"]:.4f}')
        print(f'    Mean score   = {res["scores"].mean():.1f} / 100  '
              f'({score_to_stars(res["scores"].mean())} / 5.0 stars)  [{band}]')
        print(f'    Per-class accuracy:')
        for r, acc in res['per_class'].items():
            print(f'      Rating {r}: {acc:.4f}')

    print('\n  Generating charts …')
    plot_training_curves(histories)
    plot_comparison(results, times)
    plot_confusion_matrices(results)
    plot_score_distributions(results)
    plot_score_by_rating(results, y_te)
    plot_per_class_accuracy(results)

    rows = []
    for name in ['lstm', 'gru', 'transformer']:
        r = results[name]
        band, _, stars = score_to_label(r['scores'].mean())
        rows.append({
            'Model':            LABELS[name],
            'F1 (weighted)':    r['f1'],
            'AUC-ROC':          r['auc'],
            'Train time (s)':   times[name],
            'Mean score /100':  round(float(r['scores'].mean()), 1),
            'Stars /5':         score_to_stars(r['scores'].mean()),
            'Band':             band,
        })
    pd.DataFrame(rows).to_csv('outputs/results.csv', index=False)
    print('\n  Saved → outputs/results.csv')

    best = max(results, key=lambda n: results[n]['f1'])
    with open('models/best_model.txt', 'w') as f:
        f.write(best)
    print(f'\n  Best model : {LABELS[best]}  (F1 = {results[best]["f1"]:.4f})')

    print(f'\n  {"Model":<14} {"F1":>8} {"AUC":>8} {"Time(s)":>10} '
          f'{"Score/100":>10} {"Stars/5":>8} {"Band":>12}')
    print(f'  {"─"*72}')
    for name in ['lstm', 'gru', 'transformer']:
        r = results[name]
        band, _, stars = score_to_label(r['scores'].mean())
        print(f'  {LABELS[name]:<14} {r["f1"]:>8.4f} {r["auc"]:>8.4f} '
              f'{times[name]:>10.0f} {r["scores"].mean():>10.1f} '
              f'{score_to_stars(r["scores"].mean()):>8.1f} {band:>12}')

    print(f'\n  Next: python step3_score.py  (score a new session)')
    print(f'        streamlit run app.py      (launch dashboard)')


if __name__ == '__main__':
    main()
