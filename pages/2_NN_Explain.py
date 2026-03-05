import streamlit as st
import pandas as pd

st.set_page_config(page_title="NN Explain", layout="wide")

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
hr { border-color: rgba(255,255,255,0.10) !important; }
</style>
""", unsafe_allow_html=True)

st.title("Neural Network — Explanation")

st.markdown("""
<div class="card">
  <div style="font-size:1.1rem;font-weight:900;">Goal</div>
  <div class="small-muted" style="margin-top:8px;">
    ทำนายทิศทางราคาทองพรุ่งนี้ (UP/DOWN) โดยใช้ <b>Neural Network (MLP)</b>
    ออกแบบโครงสร้างโมเดลเองให้เหมาะสมกับ dataset
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

st.subheader("1) Dataset & Features (ใช้ชุดเดียวกับ ML)")
st.markdown("""
ใช้ dataset ที่ผ่านการเตรียมข้อมูลแล้ว (`gold_ml_dataset.csv`)  
Features: `gold_ret`, `oil_ret`, `dxy_ret`, `sp_ret`  
Target: `target` (1=UP, 0=DOWN)
""")

DATA_FILE = "gold_ml_dataset.csv"
try:
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"]).sort_values("Date")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Start", df["Date"].min().date().isoformat())
    c3.metric("End", df["Date"].max().date().isoformat())
    c4.metric("UP rate", f"{df['target'].mean():.3f}")
except Exception as e:
    st.warning(f"ไม่พบ `{DATA_FILE}` หรืออ่านไม่ได้: {e}")

st.divider()

st.subheader("2) Data Preparation (เหมือน ML)")
st.markdown("""
- merge ตาม Date (Gold + Oil + DXY + S&P500)
- จัดการ missing: ffill / dropna
- สร้าง target จากวันถัดไป
- สร้าง returns
- split แบบ time-series (80/20)
- scaling ด้วย StandardScaler
""")

st.divider()

st.subheader("3) Neural Network Architecture (MLP)")
st.markdown("""
ตัวอย่างโครงสร้างที่ใช้ (Binary classification):
- Input: 4 features
- Dense(32, ReLU)
- Dense(16, ReLU)
- Dense(1, Sigmoid)

**Sigmoid output** ให้ค่าเป็น probability ของ class UP (0..1)

**Loss:** Binary Cross Entropy  
**Optimizer:** Adam  
**Threshold:** 0.5 เพื่อแปลง probability → UP/DOWN
""")

st.divider()

st.subheader("4) Evaluation (Neural Network)")
st.markdown("ค่าจากการทดสอบ (Test set):")

# จากรูปที่คุณเคยมี (ถ้าค่าเปลี่ยนค่อยแก้ตัวเลขได้)
nn_metrics = {
    "Accuracy": 0.5270270270270270,
    "Precision": 0.6138613861386139,
    "Recall": 0.484375,
    "F1": 0.5414847161572053,
    "ROC-AUC": 0.5137965425531915,
}
cm = [[55, 39],
      [66, 62]]

colA, colB, colC, colD, colE = st.columns(5)
colA.metric("Accuracy", f"{nn_metrics['Accuracy']:.3f}")
colB.metric("Precision", f"{nn_metrics['Precision']:.3f}")
colC.metric("Recall", f"{nn_metrics['Recall']:.3f}")
colD.metric("F1", f"{nn_metrics['F1']:.3f}")
colE.metric("ROC-AUC", f"{nn_metrics['ROC-AUC']:.3f}")

st.markdown("**Confusion Matrix (Actual rows x Pred columns)**")
cm_df = pd.DataFrame(cm, columns=["Pred DOWN", "Pred UP"], index=["Actual DOWN", "Actual UP"])
st.table(cm_df)

st.divider()

st.subheader("5) References")
st.markdown("""
- Investing.com Historical Data
- TensorFlow / Keras: Dense layers, Binary Cross Entropy, Adam
""")