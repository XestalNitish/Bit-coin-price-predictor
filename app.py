import streamlit as st
import numpy as np
import joblib
import json
from datetime import datetime

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Predictor | x-estal nitish",
    page_icon="₿",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ─── Import Google Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ─── Global ─── */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 40%, #0f0a1a 100%);
    }

    /* ─── Hide default header/footer ─── */
    #MainMenu, footer, header {visibility: hidden;}

    /* ─── Hero Banner ─── */
    .hero-container {
        background: linear-gradient(135deg, rgba(247,147,26,0.08) 0%, rgba(247,147,26,0.02) 100%);
        border: 1px solid rgba(247,147,26,0.15);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(247,147,26,0.03) 0%, transparent 70%);
        animation: pulse-glow 4s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }

    .hero-icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        display: block;
        filter: drop-shadow(0 0 20px rgba(247,147,26,0.4));
    }
    .hero-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #F7931A 0%, #FFD93D 50%, #F7931A 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        margin: 0;
        letter-spacing: 2px;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        color: #8b949e;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 300;
        letter-spacing: 1px;
    }

    /* ─── Author Badge ─── */
    .author-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(247,147,26,0.12) 0%, rgba(255,217,61,0.08) 100%);
        border: 1px solid rgba(247,147,26,0.25);
        border-radius: 50px;
        padding: 8px 20px;
        margin-top: 1rem;
        font-family: 'Orbitron', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        color: #F7931A;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .author-badge .dot {
        width: 8px;
        height: 8px;
        background: #F7931A;
        border-radius: 50%;
        animation: blink 1.5s ease-in-out infinite;
        box-shadow: 0 0 8px rgba(247,147,26,0.6);
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ─── Section Cards ─── */
    .section-card {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .section-card:hover {
        border-color: rgba(247,147,26,0.3);
        box-shadow: 0 4px 24px rgba(247,147,26,0.06);
    }
    .section-title {
        font-family: 'Orbitron', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: #F7931A;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(247,147,26,0.15);
    }

    /* ─── Metric Boxes ─── */
    .metric-row {
        display: flex;
        gap: 12px;
        margin-top: 1rem;
    }
    .metric-box {
        flex: 1;
        background: rgba(247,147,26,0.05);
        border: 1px solid rgba(247,147,26,0.12);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 500;
        color: #e6edf3;
    }

    /* ─── Prediction Result ─── */
    .prediction-box {
        background: linear-gradient(135deg, rgba(247,147,26,0.1) 0%, rgba(247,147,26,0.03) 100%);
        border: 2px solid rgba(247,147,26,0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .prediction-box::after {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(247,147,26,0.05), transparent);
        animation: sweep 3s ease-in-out infinite;
    }
    @keyframes sweep {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    .prediction-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .prediction-price {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #F7931A, #FFD93D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .prediction-currency {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #F7931A;
        margin-top: 0.3rem;
        letter-spacing: 1px;
    }

    /* ─── Styled Inputs ─── */
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid rgba(48, 54, 61, 0.8) !important;
        border-radius: 10px !important;
        color: #e6edf3 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stNumberInput > div > div > input:focus,
    .stDateInput > div > div > input:focus {
        border-color: rgba(247,147,26,0.5) !important;
        box-shadow: 0 0 0 2px rgba(247,147,26,0.1) !important;
    }
    .stSlider > div > div > div > div {
        background-color: #F7931A !important;
    }

    /* ─── Button ─── */
    .stButton > button {
        background: linear-gradient(135deg, #F7931A 0%, #e8850f 100%) !important;
        color: #000 !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 2px !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(247,147,26,0.35) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        border-top: 1px solid rgba(48, 54, 61, 0.4);
        margin-top: 2rem;
    }
    .footer-brand {
        font-family: 'Orbitron', monospace;
        font-size: 0.9rem;
        font-weight: 700;
        color: #F7931A;
        letter-spacing: 3px;
    }
    .footer-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #484f58;
        margin-top: 0.3rem;
        letter-spacing: 1px;
    }

    /* ─── Expander ─── */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #8b949e !important;
    }

    /* ─── Tab Styling ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ── Pure NumPy ANN Inference ─────────────────────────────────────────────────
def relu(x):
    return np.maximum(0, x)

def predict_numpy(weights_data, config, input_array):
    """Run ANN forward pass using pure NumPy — no TensorFlow needed!"""
    x = input_array.astype(np.float32)
    num_layers = len(config)
    for i in range(num_layers):
        w_key = f"w{i}"
        b_key = f"b{i}"
        if w_key in weights_data and b_key in weights_data:
            w = weights_data[w_key]
            b = weights_data[b_key]
            x = x @ w + b
            if config[i]["activation"] == "relu":
                x = relu(x)
    return x

# ── Load Artifacts (cached) ─────────────────────────────────────────────────
@st.cache_resource
def load_model_weights():
    data = np.load("btc_model_weights.npz")
    return dict(data)

@st.cache_resource
def load_model_config():
    with open("model_config.json") as f:
        return json.load(f)

@st.cache_resource
def load_scalers():
    scaler_X = joblib.load("scaler_X.joblib")
    scaler_y = joblib.load("scaler_y.joblib")
    return scaler_X, scaler_y

@st.cache_resource
def load_features():
    with open("features.json") as f:
        return json.load(f)

weights_data = load_model_weights()
model_config = load_model_config()
scaler_X, scaler_y = load_scalers()
features = load_features()

# ── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <span class="hero-icon">₿</span>
    <h1 class="hero-title">Bitcoin Price Predictor</h1>
    <p class="hero-subtitle">AI-Powered Close Price Forecasting using Deep Neural Networks</p>
    <div class="author-badge">
        <span class="dot"></span>
        x-estal nitish
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Layout: Two Columns ─────────────────────────────────────────────────
col_left, col_spacer, col_right = st.columns([5, 0.5, 5])

# ── LEFT: Market Data ───────────────────────────────────────────────────────
with col_left:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📈 Market Data</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        open_price = st.number_input(
            "Open Price (USD)",
            min_value=0.0,
            value=30000.0,
            step=100.0,
            format="%.2f",
        )
        low_price = st.number_input(
            "Low Price (USD)",
            min_value=0.0,
            value=29000.0,
            step=100.0,
            format="%.2f",
        )
    with m2:
        high_price = st.number_input(
            "High Price (USD)",
            min_value=0.0,
            value=31000.0,
            step=100.0,
            format="%.2f",
        )
        volume = st.number_input(
            "Volume",
            min_value=0.0,
            value=10.0,
            step=0.5,
            format="%.2f",
        )

# ── RIGHT: Date & Time ──────────────────────────────────────────────────────
with col_right:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📅 Date & Time</div>
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        input_date = st.date_input("Select Date", value=datetime.today())
    with d2:
        input_hour = st.slider("Hour (0-23)", min_value=0, max_value=23, value=12)

    # Derived features
    day = input_date.day
    weekday = input_date.weekday()
    month = input_date.month
    is_weekend = 1 if weekday >= 5 else 0
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-label">Day</div>
            <div class="metric-value">{day}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Weekday</div>
            <div class="metric-value">{weekday_names[weekday]}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Month</div>
            <div class="metric-value">{month_names[month]}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Weekend</div>
            <div class="metric-value">{"✅ Yes" if is_weekend else "❌ No"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Predict Button ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

btn_col1, btn_col2, btn_col3 = st.columns([2, 3, 2])
with btn_col2:
    predict_clicked = st.button("⚡ PREDICT CLOSE PRICE", use_container_width=True, type="primary")

# ── Prediction ───────────────────────────────────────────────────────────────
if predict_clicked:
    with st.spinner("🧠 Running Neural Network..."):
        # Build feature array
        input_array = np.array([[open_price, high_price, low_price, volume,
                                  input_hour, day, weekday, month, is_weekend]])

        # Scale → Predict (NumPy) → Inverse-scale
        input_scaled = scaler_X.transform(input_array)
        pred_scaled = predict_numpy(weights_data, model_config, input_scaled)
        pred_price = scaler_y.inverse_transform(pred_scaled)[0][0]

    # Result
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">Predicted BTC Close Price</div>
        <p class="prediction-price">${pred_price:,.2f}</p>
        <div class="prediction-currency">United States Dollar (USD)</div>
    </div>
    """, unsafe_allow_html=True)

    # Input summary
    with st.expander("🔍 View Prediction Details"):
        det1, det2 = st.columns(2)
        with det1:
            st.markdown("**Input Features:**")
            input_dict = dict(zip(features, input_array[0]))
            for k, v in input_dict.items():
                st.code(f"{k}: {v}", language=None)
        with det2:
            st.markdown("**Model Info:**")
            st.code(f"Scaled prediction: {pred_scaled[0][0]:.6f}", language=None)
            st.code(f"Unscaled price: ${pred_price:,.2f}", language=None)
            st.code(f"Features used: {len(features)}", language=None)
            st.code(f"Model params: ~48,131 trainable", language=None)
            st.code(f"Engine: Pure NumPy (no TF needed)", language=None)

# ── Model Info Section ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("🧠 About the Model"):
    info1, info2 = st.columns(2)
    with info1:
        st.markdown("""
        **Architecture:** Sequential ANN  
        **Layers:** 9 Dense layers  
        **Activation:** ReLU (hidden), Linear (output)  
        **Parameters:** ~48,131 trainable  
        """)
    with info2:
        st.markdown("""
        **Scaling:** StandardScaler (scikit-learn)  
        **Input Features:** 9  
        **Output:** Close Price (USD)  
        **Inference:** Pure NumPy (lightweight)  
        """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-brand">⚡ x-estal nitish</div>
    <div class="footer-text">Built with Streamlit • ANN Model • Deep Learning Project</div>
</div>
""", unsafe_allow_html=True)
