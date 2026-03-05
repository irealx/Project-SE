import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="NN Test", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.block-container { padding-top: 2.2rem; padding-bottom: 2.5rem; }
.stApp {
  background: radial-gradient(1200px circle at 10% 0%, rgba(99,102,241,0.12), transparent 45%),
              radial-gradient(900px circle at 90% 10%, rgba(34,197,94,0.10), transparent 45%),
              #0b1220;
  color: #e5e7eb;
}
h1, h2, h3, .stMarkdown { color: #e5e7eb; }
.small-muted { color: #9ca3af; font-size: 0.92rem; }
.card {
  background: rgba(17, 24, 39, 0.72);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
div[data-baseweb="input"] > div { background: rgba(17,24,39,0.85) !important; border-radius: 12px !important; }
.stButton > button {
  background: linear-gradient(135deg, #6366f1, #22c55e);
  color: white; border: 0; border-radius: 12px; padding: 0.65rem 1rem; font-weight: 800;
}
.stProgress > div > div > div > div { background: linear-gradient(90deg, #22c55e, #6366f1) !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.pkl")
    nn = tf.keras.models.load_model("nn.keras")
    return scaler, nn

scaler, nn = load_assets()

def safe_return(today: float, yesterday: float) -> float:
    if yesterday == 0:
        return np.nan
    return (today - yesterday) / yesterday

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def pct(x: float) -> str:
    return f"{x*100:.3f}%"

st.title("Test — Neural Network")

st.markdown("""
<div class="card">
  <div style="font-size:1.1rem;font-weight:900;">NN Test Page</div>
  <div class="small-muted" style="margin-top:8px;">
    ทดสอบเฉพาะ <b>Neural Network (MLP)</b> และแสดง Probability ของ <b>UP</b>
  </div>
</div>
""", unsafe_allow_html=True)

mode = st.radio("Input mode", ["Price mode (Yesterday vs Today)", "Return mode (direct %)"], horizontal=True)

colA, colB = st.columns([1,1])
use_example = colA.button("Use example")
reset = colB.button("Reset")

EX = {
    "gold_y": 2000.0, "gold_t": 2005.0,
    "oil_y": 80.0, "oil_t": 80.5,
    "dxy_y": 105.0, "dxy_t": 105.2,
    "sp_y": 5000.0, "sp_t": 5010.0,
    "gold_r": 0.25, "oil_r": 0.30, "dxy_r": 0.10, "sp_r": 0.20
}

if reset:
    for k in list(EX.keys()):
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

if use_example:
    for k, v in EX.items():
        st.session_state[k] = v
    st.rerun()

st.divider()

errs = []
gold_ret = oil_ret = dxy_ret = sp_ret = np.nan

if mode.startswith("Price mode"):
    with st.expander("Gold (GC) — USD/oz", expanded=True):
        gold_y = st.number_input("Gold yesterday (USD/oz)", key="gold_y", value=st.session_state.get("gold_y", EX["gold_y"]), step=1.0, format="%.2f")
        gold_t = st.number_input("Gold today (USD/oz)", key="gold_t", value=st.session_state.get("gold_t", EX["gold_t"]), step=1.0, format="%.2f")

    with st.expander("Crude Oil WTI (CL) — USD/bbl", expanded=True):
        oil_y = st.number_input("Oil yesterday (USD/bbl)", key="oil_y", value=st.session_state.get("oil_y", EX["oil_y"]), step=0.1, format="%.2f")
        oil_t = st.number_input("Oil today (USD/bbl)", key="oil_t", value=st.session_state.get("oil_t", EX["oil_t"]), step=0.1, format="%.2f")

    with st.expander("US Dollar Index (DXY) — index", expanded=True):
        dxy_y = st.number_input("DXY yesterday (index)", key="dxy_y", value=st.session_state.get("dxy_y", EX["dxy_y"]), step=0.1, format="%.2f")
        dxy_t = st.number_input("DXY today (index)", key="dxy_t", value=st.session_state.get("dxy_t", EX["dxy_t"]), step=0.1, format="%.2f")

    with st.expander("S&P500 (US500) — index", expanded=True):
        sp_y = st.number_input("S&P500 yesterday (index)", key="sp_y", value=st.session_state.get("sp_y", EX["sp_y"]), step=1.0, format="%.2f")
        sp_t = st.number_input("S&P500 today (index)", key="sp_t", value=st.session_state.get("sp_t", EX["sp_t"]), step=1.0, format="%.2f")

    if min(gold_y, gold_t, oil_y, oil_t, dxy_y, dxy_t, sp_y, sp_t) <= 0:
        errs.append("All prices must be > 0")

    gold_ret = safe_return(gold_t, gold_y)
    oil_ret  = safe_return(oil_t, oil_y)
    dxy_ret  = safe_return(dxy_t, dxy_y)
    sp_ret   = safe_return(sp_t, sp_y)

else:
    c1, c2 = st.columns(2)
    with c1:
        gold_r = st.number_input("Gold daily return (%)", key="gold_r", value=st.session_state.get("gold_r", EX["gold_r"]), step=0.05, format="%.3f")
        oil_r  = st.number_input("Oil daily return (%)", key="oil_r", value=st.session_state.get("oil_r", EX["oil_r"]), step=0.05, format="%.3f")
    with c2:
        dxy_r  = st.number_input("DXY daily return (%)", key="dxy_r", value=st.session_state.get("dxy_r", EX["dxy_r"]), step=0.02, format="%.3f")
        sp_r   = st.number_input("S&P500 daily return (%)", key="sp_r", value=st.session_state.get("sp_r", EX["sp_r"]), step=0.05, format="%.3f")

    gold_ret = gold_r / 100.0
    oil_ret  = oil_r / 100.0
    dxy_ret  = dxy_r / 100.0
    sp_ret   = sp_r / 100.0

X_raw = np.array([[gold_ret, oil_ret, dxy_ret, sp_ret]], dtype=float)

if np.any(~np.isfinite(X_raw)):
    errs.append("Invalid returns (check inputs).")

if not errs:
    X_raw[0, 0] = clamp(float(X_raw[0, 0]), -0.20, 0.20)
    X_raw[0, 1] = clamp(float(X_raw[0, 1]), -0.20, 0.20)
    X_raw[0, 2] = clamp(float(X_raw[0, 2]), -0.10, 0.10)
    X_raw[0, 3] = clamp(float(X_raw[0, 3]), -0.20, 0.20)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Computed features (returns)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("gold_ret", pct(float(X_raw[0,0])) if np.isfinite(X_raw[0,0]) else "—")
m2.metric("oil_ret",  pct(float(X_raw[0,1])) if np.isfinite(X_raw[0,1]) else "—")
m3.metric("dxy_ret",  pct(float(X_raw[0,2])) if np.isfinite(X_raw[0,2]) else "—")
m4.metric("sp_ret",   pct(float(X_raw[0,3])) if np.isfinite(X_raw[0,3]) else "—")
st.markdown("</div>", unsafe_allow_html=True)

if errs:
    st.error("Please fix:")
    for e in errs:
        st.write(f"- {e}")

btn = st.button("Predict (NN)", disabled=bool(errs))

if btn and not errs:
    X = scaler.transform(X_raw)
    proba_up = float(nn.predict(X, verbose=0).ravel()[0])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("NN Prediction Result")
    st.progress(proba_up)
    st.write(f"**P(UP)** = `{proba_up*100:.1f}%`")

    if proba_up >= 0.5:
        st.success("Prediction: **UP tomorrow**")
    else:
        st.error("Prediction: **DOWN tomorrow**")

    st.caption("This page tests only the Neural Network model.")
    st.markdown("</div>", unsafe_allow_html=True)