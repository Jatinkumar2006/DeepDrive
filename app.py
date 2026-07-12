"""
app.py  —  Deep Learning Driving Score Dashboard
=================================================
Upload any raw session CSV -> driving score from LSTM, GRU, or Transformer.

SCORE DIRECTION:
  100 = perfect / safest driver  (Rating 5)
  0   = most dangerous driver    (Rating 1)

Run:  streamlit run app.py
"""

# DeepDrive Streamlit Dashboard
# This module provides a highly interactive web interface for the trained deep learning models.
# It features an advanced UI architecture using custom CSS glassmorphism and interactive Plotly visualizations.
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

WINDOW      = 100
STEP        = 50
SENSOR_COLS = ['X_Acc', 'Y_Acc', 'Z_Acc', 'X_Gyro', 'Y_Gyro', 'Z_Gyro']
ACC_LIMIT   = 40.0
GYRO_LIMIT  = 300.0
GAP_MS      = 5000

# 100 = safest (Rating 5), 0 = most dangerous (Rating 1)
RISK_W       = np.array([0, 25, 50, 75, 100], dtype=np.float32)
MODEL_COLORS = {'lstm': '#3B82F6', 'gru': '#06B6D4', 'transformer': '#F59E0B'}
MODEL_LABELS = {'lstm': 'LSTM', 'gru': 'GRU', 'transformer': 'Transformer'}


def score_to_label(score):
    """100=perfect, 0=dangerous."""
    if score >= 80: return 'Excellent',  '#22C55E', '★★★★★'
    if score >= 60: return 'Good',       '#84CC16', '★★★★☆'
    if score >= 40: return 'Moderate',   '#F59E0B', '★★★☆☆'
    if score >= 20: return 'Poor',       '#EF4444', '★★☆☆☆'
    return                 'Dangerous',  '#EF4444', '★☆☆☆☆' # Bright red for neon glow


def score_to_stars(score):
    return round(score / 20, 1)


RECOMMENDATIONS = {
    'Excellent':  '✅ Excellent driving! Smooth, controlled, and consistent throughout.',
    'Good':       '👍 Good driving overall. Minor improvements possible in cornering or braking.',
    'Moderate':   '⚡ Moderate driving. Some aggressive events detected — try smoother acceleration.',
    'Poor':       '⚠️ Poor driving detected. Reduce speed and avoid sudden braking or sharp turns.',
    'Dangerous':  '🚨 Dangerous driving patterns. Immediate improvement needed for safety.',
}


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


@st.cache_resource
def load_all_models(): # Cache busted
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        return {}, None

    TransformerBlock = _make_transformer_block(tf)
    models = {}
    for name in ['lstm', 'gru', 'transformer']:
        path = f'models/{name}_model.h5'
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(
                path, custom_objects={'TransformerBlock': TransformerBlock})
            
    scaler = None
    if os.path.exists('data/dl_scaler.pkl'):
        scaler = joblib.load('data/dl_scaler.pkl')

    return models, scaler


def clean_upload(df):
    drop = [c for c in df.columns if c.strip().upper() in ('ID', 'NAME')]
    df = df.drop(columns=drop, errors='ignore')
    if 'Rating' not in df.columns: df['Rating'] = 3
    df[SENSOR_COLS] = df[SENSOR_COLS].apply(pd.to_numeric, errors='coerce')
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill().bfill()
    df = df.dropna(subset=SENSOR_COLS)
    df = df[~(df[SENSOR_COLS] == 0).all(axis=1)] # type: ignore
    df = df[df['Z_Acc'] != 0]
    for col in ['X_Acc', 'Y_Acc', 'Z_Acc']: df[col] = df[col].clip(-ACC_LIMIT, ACC_LIMIT)
    for col in ['X_Gyro', 'Y_Gyro', 'Z_Gyro']: df[col] = df[col].clip(-GYRO_LIMIT, GYRO_LIMIT)
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    df['session_id'] = (df['Timestamp'].diff().fillna(0) > GAP_MS).cumsum()
    return df.reset_index(drop=True)


def make_windows(df, scaler):
    mu, sig = scaler['mean'], scaler['std']
    windows = []
    for _, sdf in df.groupby('session_id'):
        sdf = sdf.reset_index(drop=True)
        vals = sdf[SENSOR_COLS].values.astype(np.float32)
        for i in range(0, len(sdf) - WINDOW + 1, STEP):
            windows.append(vals[i: i + WINDOW])
    if not windows: return np.empty((0, WINDOW, 6), dtype=np.float32)
    X = np.array(windows, dtype=np.float32)
    return (X - mu) / sig


# ==========================================
# UI CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title='DeepDrive AI Evaluator',
    page_icon='🏍️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom Glassmorphism Theme CSS
st.markdown("""
<style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Outfit', sans-serif; 
    }
    
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0F172A 100%);
        color: #F8FAFC;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Hide standard top header */
    header { background: transparent !important; }

    /* Custom Glass Card */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    }
    
    /* Glowing Text Effect */
    .glow-text {
        background: linear-gradient(to right, #38BDF8, #818CF8);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Metric Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: background 0.3s ease;
    }
    .metric-box:hover {
        background: rgba(255, 255, 255, 0.06);
    }
    .metric-title {
        font-size: 14px;
        color: #94A3B8;
        margin-bottom: 8px;
        font-weight: 400;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown('<h1 style="font-size: 28px; margin-bottom: 0;">🏍️ DeepDrive</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; margin-top: -10px;">AI Driving Safety Evaluator</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("1. Configuration")
    # Cache the deeply nested Keras models in memory to ensure the UI remains highly responsive across sessions
    models, scaler = load_all_models()
    
    if not models:
        st.error('No models found. Run `python run_pipeline.py`.')
        st.stop()
    elif scaler is None:
        st.error('Scaler missing.')
        st.stop()
        
    avail_names = list(models.keys())
    model_choice = str(st.radio(
        'AI Engine',
        avail_names,
        format_func=lambda n: MODEL_LABELS.get(str(n), str(n)),
        help="Select the deep learning architecture for evaluation."
    ))
    
    st.markdown("---")
    st.subheader("2. Analyze Session")
    uploaded = st.file_uploader('Upload Sensor Data (CSV)', type=['csv'])


# ==========================================
# MAIN PAGE
# ==========================================
if not uploaded:
    # Empty State
    st.markdown("""
<div style="height: 70vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
    <h1 style="font-size: 56px; margin-bottom: 16px;" class="glow-text">Ready for Analysis</h1>
    <p style="font-size: 20px; color: #94A3B8; max-width: 600px; line-height: 1.5;">
        Upload a raw smartphone sensor CSV in the sidebar to generate a highly accurate driving safety score using deep learning.
    </p>
</div>
""", unsafe_allow_html=True)
    st.stop()

# Process Uploaded Data
raw_df = pd.read_csv(uploaded, low_memory=False)
missing = [c for c in SENSOR_COLS if c not in raw_df.columns]
if missing:
    st.error(f'Missing required columns: {missing}')
    st.stop()

with st.spinner('Extracting sequences...'):
    clean_df = clean_upload(raw_df)
    if len(clean_df) < WINDOW:
        st.error(f'Not enough rows ({len(clean_df)}) — need >= {WINDOW}')
        st.stop()
    X = make_windows(clean_df, scaler)

if len(X) == 0:
    st.error('No complete windows could be extracted.')
    st.stop()

with st.spinner(f'Running {MODEL_LABELS[model_choice]} inference...'):
    model  = models[model_choice]
    proba  = model.predict(X, verbose=0)
    scores = (proba * RISK_W).sum(axis=1)
    score  = float(scores.mean())

label, color, stars = score_to_label(score)
star_num = score_to_stars(score)

# ==========================================
# DASHBOARD RENDERING
# ==========================================

# 1. Main Glowing Banner
banner_html = f"""
<div class="glass-card" style="display: flex; align-items: center; justify-content: space-between; position: relative; overflow: hidden;">
<!-- Ambient Glow -->
<div style="position: absolute; right: -50px; top: -50px; width: 250px; height: 250px; background: {color}; filter: blur(100px); opacity: 0.15; border-radius: 50%;"></div>
<div>
<div style="font-size: 16px; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Overall Safety Score</div>
<div style="display: flex; align-items: baseline; gap: 16px;">
<div style="font-size: 72px; font-weight: 700; color: #F8FAFC; line-height: 1;">{score:.1f}</div>
<div style="font-size: 24px; color: {color}; font-weight: 600;">{label}</div>
</div>
<div style="margin-top: 12px; color: #CBD5E1; font-size: 18px;">{RECOMMENDATIONS[label]}</div>
</div>
<div style="text-align: right; z-index: 1;">
<div style="font-size: 40px; margin-bottom: 8px;">{stars}</div>
<div style="font-size: 18px; color: #94A3B8;">{star_num} / 5.0 Stars</div>
<div style="font-size: 14px; color: #64748B; margin-top: 4px;">Analyzed by {MODEL_LABELS[model_choice]}</div>
</div>
</div>
"""
st.markdown(banner_html, unsafe_allow_html=True)


# 2. Interactive Plotly Timeline
st.markdown('<h3 style="margin-bottom: 16px;">Driving Session Timeline</h3>', unsafe_allow_html=True)

fig = go.Figure()

# Background Zones
fig.add_hrect(y0=80, y1=100, fillcolor="#22C55E", opacity=0.05, line_width=0)
fig.add_hrect(y0=60, y1=80, fillcolor="#84CC16", opacity=0.05, line_width=0)
fig.add_hrect(y0=40, y1=60, fillcolor="#F59E0B", opacity=0.05, line_width=0)
fig.add_hrect(y0=20, y1=40, fillcolor="#EF4444", opacity=0.05, line_width=0)
fig.add_hrect(y0=0, y1=20, fillcolor="#991B1B", opacity=0.05, line_width=0)

# Main Area Chart
fig.add_trace(go.Scatter(
    y=scores,
    x=list(range(len(scores))),
    fill='tozeroy',
    mode='lines',
    line=dict(color=color, width=3),
    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
    name='Driving Score',
    hovertemplate="Window: %{x}<br>Score: %{y:.1f}<extra></extra>"
))

# Styling
fig.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    height=300,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=False, title="Window (~2s elapsed)", color="#94A3B8"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", range=[0, 105], color="#94A3B8"),
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)
# 3. Metrics Grid
metrics = [
    ('Excellent', int((scores >= 80).sum()), '#22C55E'),
    ('Good', int(((scores >= 60) & (scores < 80)).sum()), '#84CC16'),
    ('Moderate', int(((scores >= 40) & (scores < 60)).sum()), '#F59E0B'),
    ('Poor', int(((scores >= 20) & (scores < 40)).sum()), '#EF4444'),
    ('Dangerous', int((scores < 20).sum()), '#991B1B'),
]

grid_html = """
<div class="glass-card">
<h3 style="margin-top: 0;">Session Breakdown</h3>
<div class="metric-grid">
"""
for title, count, c_hex in metrics:
    pct = (count / len(scores)) * 100
    grid_html += f"""
<div class="metric-box" style="border-top: 3px solid {c_hex};">
<div class="metric-title">{title}</div>
<div class="metric-value">{count}</div>
<div style="color: #64748B; font-size: 13px; margin-top: 4px;">{pct:.1f}%</div>
</div>
"""
grid_html += "</div></div>"
st.markdown(grid_html, unsafe_allow_html=True)


# 4. Model Comparison (Optional)
if len(models) > 1:
    with st.expander("Compare All Models"):
        st.markdown('<div style="padding: 16px;">', unsafe_allow_html=True)
        rows = []
        for name, mdl in models.items():
            if name != model_choice:
                p = mdl.predict(X, verbose=0)
                s = (p * RISK_W).sum(axis=1)
                lbl, c, sts = score_to_label(float(s.mean()))
                rows.append({
                    'Model': MODEL_LABELS[name],
                    'Score': f'{s.mean():.1f}',
                    'Rating': sts,
                    'Band': lbl
                })
            else:
                rows.append({
                    'Model': MODEL_LABELS[name] + " (Active)",
                    'Score': f'{score:.1f}',
                    'Rating': stars,
                    'Band': label
                })
        
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
