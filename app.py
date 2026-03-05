import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="Gold Direction Prediction", layout="centered")

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.pkl")
    ensemble = joblib.load("ensemble.pkl")
    nn = tf.keras.models.load_model("nn.keras")
    return scaler, ensemble, nn

scaler, ensemble, nn = load_assets()

# -----------------------------
# Helpers
# -----------------------------
def safe_return(today: float, yesterday: float) -> float:
    """Return as ratio (0.01 = +1%)."""
    if yesterday == 0:
        return np.nan
    return (today - yesterday) / yesterday

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def validate_price_pair(yesterday: float, today: float, name: str) -> list[str]:
    errs = []
    if yesterday <= 0 or today <= 0:
        errs.append(f"{name}: prices must be > 0")
    # Prevent accidental huge numbers (optional)
    if yesterday > 1e9 or today > 1e9:
        errs.append(f"{name}: price looks unrealistically large")
    return errs

def predict_proba_up(X_raw: np.ndarray):
    """X_raw is shape (1,4) in return-ratio units."""
    X = scaler.transform(X_raw)
    proba_ml = float(ensemble.predict_proba(X)[:, 1][0])
    proba_nn = float(nn.predict(X, verbose=0).ravel()[0])
    proba = (proba_ml + proba_nn) / 2.0
    return proba, proba_ml, proba_nn

def pct(x: float) -> str:
    return f"{x*100:.3f}%"


# -----------------------------
# UI Header
# -----------------------------
st.title("Gold Price Direction Prediction")
st.caption("Predict whether gold price will go **UP** or **DOWN** tomorrow using ML Ensemble + Neural Network.")

# Mode selector
mode = st.radio(
    "Input mode",
    ["Price mode (Yesterday vs Today)", "Return mode (direct %)"],
    horizontal=True
)

st.divider()

# Example button
if "example_loaded" not in st.session_state:
    st.session_state.example_loaded = False

col_ex1, col_ex2 = st.columns([1, 2])
with col_ex1:
    load_example = st.button("Use example values")
with col_ex2:
    st.write("Tip: Use example values to test the app quickly.")

# Default/example values
EX = {
    "gold_y": 2000.0, "gold_t": 2005.0,
    "oil_y": 80.0, "oil_t": 80.5,
    "dxy_y": 105.0, "dxy_t": 105.2,
    "sp_y": 5000.0, "sp_t": 5010.0,
    # return mode example (%)
    "gold_r": 0.25, "oil_r": 0.30, "dxy_r": 0.10, "sp_r": 0.20
}

if load_example:
    st.session_state.example_loaded = True
    for k, v in EX.items():
        st.session_state[k] = v

# -----------------------------
# INPUTS
# -----------------------------
errs = []
X_raw = None  # final (1,4) return ratios

if mode.startswith("Price mode"):
    st.subheader("Enter prices (Yesterday vs Today)")
    st.write("The app will compute daily returns automatically.")

    with st.expander("Gold (GC)", expanded=True):
        gold_y = st.number_input("Gold yesterday", key="gold_y", value=st.session_state.get("gold_y", EX["gold_y"]), step=1.0, format="%.2f")
        gold_t = st.number_input("Gold today", key="gold_t", value=st.session_state.get("gold_t", EX["gold_t"]), step=1.0, format="%.2f")

    with st.expander("Crude Oil WTI (CL)", expanded=True):
        oil_y = st.number_input("Oil yesterday", key="oil_y", value=st.session_state.get("oil_y", EX["oil_y"]), step=0.1, format="%.2f")
        oil_t = st.number_input("Oil today", key="oil_t", value=st.session_state.get("oil_t", EX["oil_t"]), step=0.1, format="%.2f")

    with st.expander("US Dollar Index (DXY)", expanded=True):
        dxy_y = st.number_input("DXY yesterday", key="dxy_y", value=st.session_state.get("dxy_y", EX["dxy_y"]), step=0.1, format="%.2f")
        dxy_t = st.number_input("DXY today", key="dxy_t", value=st.session_state.get("dxy_t", EX["dxy_t"]), step=0.1, format="%.2f")

    with st.expander("S&P500 (US500)", expanded=True):
        sp_y = st.number_input("S&P500 yesterday", key="sp_y", value=st.session_state.get("sp_y", EX["sp_y"]), step=1.0, format="%.2f")
        sp_t = st.number_input("S&P500 today", key="sp_t", value=st.session_state.get("sp_t", EX["sp_t"]), step=1.0, format="%.2f")

    # validate
    errs += validate_price_pair(gold_y, gold_t, "Gold")
    errs += validate_price_pair(oil_y, oil_t, "Oil")
    errs += validate_price_pair(dxy_y, dxy_t, "DXY")
    errs += validate_price_pair(sp_y, sp_t, "S&P500")

    gold_ret = safe_return(gold_t, gold_y)
    oil_ret = safe_return(oil_t, oil_y)
    dxy_ret = safe_return(dxy_t, dxy_y)
    sp_ret = safe_return(sp_t, sp_y)

    # show computed returns
    st.subheader("Computed returns")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("gold_ret", pct(gold_ret) if np.isfinite(gold_ret) else "—")
    c2.metric("oil_ret", pct(oil_ret) if np.isfinite(oil_ret) else "—")
    c3.metric("dxy_ret", pct(dxy_ret) if np.isfinite(dxy_ret) else "—")
    c4.metric("sp_ret", pct(sp_ret) if np.isfinite(sp_ret) else "—")

    X_raw = np.array([[gold_ret, oil_ret, dxy_ret, sp_ret]], dtype=float)

else:
    st.subheader("Enter returns directly (%)")
    st.write("Example: enter **1.25** for +1.25% (the app converts to ratio internally).")

    c1, c2 = st.columns(2)
    with c1:
        gold_r = st.number_input("Gold return (%)", key="gold_r", value=st.session_state.get("gold_r", EX["gold_r"]), step=0.05, format="%.3f")
        oil_r  = st.number_input("Oil return (%)", key="oil_r", value=st.session_state.get("oil_r", EX["oil_r"]), step=0.05, format="%.3f")
    with c2:
        dxy_r  = st.number_input("DXY return (%)", key="dxy_r", value=st.session_state.get("dxy_r", EX["dxy_r"]), step=0.02, format="%.3f")
        sp_r   = st.number_input("S&P500 return (%)", key="sp_r", value=st.session_state.get("sp_r", EX["sp_r"]), step=0.05, format="%.3f")

    # convert % to ratio
    gold_ret = gold_r / 100.0
    oil_ret  = oil_r / 100.0
    dxy_ret  = dxy_r / 100.0
    sp_ret   = sp_r / 100.0

    st.subheader("Returns (ratio used by model)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("gold_ret", pct(gold_ret))
    c2.metric("oil_ret", pct(oil_ret))
    c3.metric("dxy_ret", pct(dxy_ret))
    c4.metric("sp_ret", pct(sp_ret))

    X_raw = np.array([[gold_ret, oil_ret, dxy_ret, sp_ret]], dtype=float)

# Additional validation for returns
if X_raw is not None and np.any(~np.isfinite(X_raw)):
    errs.append("Some computed returns are invalid (check inputs).")

# Clamp extreme values to avoid accidental input explosions
# Typical daily ranges (rough): Gold/Oil/SP500 within ±20%, DXY within ±10%
if X_raw is not None and len(errs) == 0:
    X_raw[0, 0] = clamp(float(X_raw[0, 0]), -0.20, 0.20)
    X_raw[0, 1] = clamp(float(X_raw[0, 1]), -0.20, 0.20)
    X_raw[0, 2] = clamp(float(X_raw[0, 2]), -0.10, 0.10)
    X_raw[0, 3] = clamp(float(X_raw[0, 3]), -0.20, 0.20)

st.divider()

# Show validation errors if any
if errs:
    st.error("Please fix the following issues:")
    for e in errs:
        st.write(f"- {e}")

# -----------------------------
# Predict
# -----------------------------
btn = st.button("Predict", disabled=bool(errs))

if btn and X_raw is not None:
    proba, proba_ml, proba_nn = predict_proba_up(X_raw)

    st.subheader("Prediction Result")

    # Probability bar (0..1)
    st.progress(float(proba))
    st.write(f"**Final probability (UP)** = `{proba:.3f}`")

    if proba >= 0.5:
        st.success("Prediction: **UP tomorrow**")
    else:
        st.error("Prediction: **DOWN tomorrow**")

    with st.expander("Model breakdown (details)"):
        st.write(f"Ensemble (Voting) P(UP): `{proba_ml:.3f}`")
        st.write(f"Neural Network P(UP): `{proba_nn:.3f}`")
        st.write("Final = average of Ensemble and NN")

    st.info("Note: This is a prediction from historical patterns. Financial markets are noisy and predictions can be wrong.")

st.divider()
with st.expander("How this app works (quick explanation)"):
    st.markdown(
        """
- You provide **today and yesterday prices** (or returns).
- The app computes **daily returns**:
  - `ret = (today - yesterday) / yesterday`
- Features used by the model:
  - `gold_ret`, `oil_ret`, `dxy_ret`, `sp_ret`
- Models:
  - **Machine Learning Ensemble** (Voting of LR + RF + GB)
  - **Neural Network** (MLP)
- Output:
  - UP/DOWN prediction for tomorrow with probability
        """.strip()
    )