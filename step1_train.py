"""
DeepDrive Project: Step 1 - Model Training
===========================================
Module 2 of the DeepDrive pipeline.
This script intakes the pre-processed sequence arrays from step0_prepare.py and trains 
three parallel deep learning architectures (LSTM, GRU, and Transformer) to classify driving risk.

Architectural Design Choices:
  1. EarlyStopping with restore_best_weights ensures optimal generalization without overfitting.
  2. ReduceLROnPlateau dynamically decays the learning rate to escape local minima.
  3. ModelCheckpoint explicitly guarantees the deployment of the highest validation accuracy model.
  4. Native .keras saving format ensures strict compatibility with the step2, step3, and app modules.

Input: Normalized sequence data from Step 0.
Output: Trained model binaries (.keras format) exported for Step 2 evaluation and Step 3 inference.
"""

import os
import time
import json
import warnings
import numpy as np
import joblib

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.makedirs('models', exist_ok=True)

EPOCHS    = 50      # max epochs — early stopping kicks in when needed
BATCH     = 64
N_CLASSES = 5
PATIENCE  = 8       # stop if no val_loss improvement for 8 epochs


def check_tf():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        print(f'  TensorFlow {tf.__version__}')
        return tf
    except ImportError:
        print('\n  ERROR: TensorFlow not installed.')
        print('  Activate tf_env and run:  pip install tensorflow')
        raise SystemExit(1)


# Version-safe TransformerBlock
def _make_transformer_block(tf):
    keras    = tf.keras
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

    if register is not None:
        try:
            TransformerBlock = register( # type: ignore
                package='DrivingRisk', name='TransformerBlock')(TransformerBlock)
        except Exception:
            pass
    return TransformerBlock


# Model definitions
def build_lstm(tf):
    keras = tf.keras
    return keras.Sequential([
        keras.layers.Input(shape=(100, 6)),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(N_CLASSES, activation='softmax'),
    ], name='lstm')


def build_gru(tf):
    keras = tf.keras
    return keras.Sequential([
        keras.layers.Input(shape=(100, 6)),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.GRU(64),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(N_CLASSES, activation='softmax'),
    ], name='gru')


def build_transformer(tf):
    TransformerBlock = _make_transformer_block(tf)
    keras = tf.keras
    inp = keras.Input(shape=(100, 6))
    x   = keras.layers.Dense(64)(inp)
    x   = TransformerBlock(64, heads=4, ff=128, name='attn_block_1')(x)
    x   = TransformerBlock(64, heads=4, ff=128, name='attn_block_2')(x)
    x   = keras.layers.GlobalAveragePooling1D()(x)
    x   = keras.layers.Dropout(0.3)(x)
    x   = keras.layers.Dense(64, activation='gelu')(x)
    x   = keras.layers.BatchNormalization()(x)
    out = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    return keras.Model(inp, out, name='transformer')


# Training — with smart callbacks
def train_one(tf, name, build_fn, X_tr, y_tr):
    print(f'\n  {"="*50}')
    print(f'  Training {name.upper()}  —  up to {EPOCHS} epochs')
    print(f'  (EarlyStopping patience={PATIENCE}, restores best weights)')
    print(f'  {"="*50}')

    # Shift labels down by 1 because sparse_categorical_crossentropy expects 0-indexed integer labels
    y_tr0 = y_tr - 1

    model = build_fn(tf)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    save_path = f'models/{name}_model.keras'

    # Configure dynamic callbacks to save the best weights, reduce learning rate on plateaus, and stop early if needed
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    t0   = time.time()
    hist = model.fit(
        X_tr, y_tr0,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = round(time.time() - t0, 1)

    actual_epochs = len(hist.history['loss'])
    best_val_acc  = max(hist.history['val_accuracy'])
    best_epoch    = hist.history['val_accuracy'].index(best_val_acc) + 1

    print(f'\n  {name.upper()} completed:')
    print(f'    Ran      : {actual_epochs} / {EPOCHS} epochs')
    print(f'    Best val_acc : {best_val_acc:.4f}  (epoch {best_epoch})')
    print(f'    Final train acc : {hist.history["accuracy"][-1]:.4f}')
    print(f'    Training time   : {elapsed}s')
    print(f'    Best model saved → {save_path}')

    return hist, elapsed


# Main
def main():
    print('\n' + '='*55)
    print(f'  Step 1 — Training deep learning models (up to {EPOCHS} epochs each)')
    print('='*55)

    tf = check_tf()

    for f in ['data/sequences_X.npy', 'data/sequences_y.npy',
              'data/split_indices.pkl']:
        if not os.path.exists(f):
            print(f'  ERROR: {f} not found. Run step0_prepare.py first.')
            raise SystemExit(1)

    X      = np.load('data/sequences_X.npy')
    y      = np.load('data/sequences_y.npy')
    splits = joblib.load('data/split_indices.pkl')
    tr_idx = splits['train']

    X_tr = X[tr_idx]
    y_tr = y[tr_idx]

    print(f'\n  Training windows : {len(X_tr):,}')
    print(f'  Shape            : {X_tr.shape}')
    print(f'  Max epochs       : {EPOCHS}')
    print(f'  Batch size       : {BATCH}')
    print(f'  Early stop pat.  : {PATIENCE}')
    print(f'\n  Label distribution (train):')
    for r, cnt in zip(*np.unique(y_tr, return_counts=True)):
        pct = 100 * cnt / len(y_tr)
        print(f'    Rating {r}: {cnt:>6,} ({pct:.1f}%)')

    builders  = {
        'lstm':        build_lstm,
        'gru':         build_gru,
        'transformer': build_transformer,
    }
    histories = {}
    times     = {}

    # Sequentially train all three deep learning architectures and record their histories
    for name, builder in builders.items():
        hist, elapsed = train_one(tf, name, builder, X_tr, y_tr)
        histories[name] = {
            'loss':         hist.history['loss'],
            'val_loss':     hist.history['val_loss'],
            'accuracy':     hist.history['accuracy'],
            'val_accuracy': hist.history['val_accuracy'],
        }
        times[name] = elapsed

    # Save histories and times
    joblib.dump(histories, 'models/train_history.pkl')
    with open('models/train_times.json', 'w') as f:
        json.dump(times, f, indent=2)

    # Summary table
    print(f'\n\n  {"="*60}')
    print(f'  TRAINING SUMMARY')
    print(f'  {"="*60}')
    print(f'  {"Model":<14} {"Epochs":>8} {"Best val acc":>14} {"Time (s)":>10}')
    print(f'  {"-"*50}')
    for name in ['lstm', 'gru', 'transformer']:
        h = histories[name]
        best_acc = max(h['val_accuracy'])
        print(f'  {name.upper():<14} {len(h["loss"]):>8} '
              f'{best_acc:>14.4f} '
              f'{times[name]:>10.1f}')
    print(f'  {"="*60}')
    print(f'\n  Saved → models/train_history.pkl')
    print(f'  Saved → models/train_times.json')
    print(f'\n  Each model saved at its BEST val_accuracy checkpoint.')
    print(f'  Next: python step2_evaluate.py')


if __name__ == '__main__':
    main()
