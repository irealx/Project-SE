import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML Explain", layout="wide")

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

st.title("Machine Learning (Ensemble) — Explanation")

st.markdown("""
<div class="card">
  <div style="font-size:1.1rem;font-weight:900;">Goal</div>
  <div class="small-muted" style="margin-top:8px;">
    ทำนายว่า “ราคาทอง <b>พรุ่งนี้</b>” จะ <b>UP</b> หรือ <b>DOWN</b> (Binary Classification)
    โดยใช้ Machine Learning แบบ <b>Ensemble</b> (รวมโมเดลตั้งแต่ 3 ประเภทขึ้นไป)
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------- Dataset overview ----------
st.subheader("1) Dataset & Source")
st.markdown("""
- ใช้ข้อมูล Historical Data จาก Investing.com (Download เป็น CSV)
- อย่างน้อย 2 ชุด (โปรเจกต์นี้ใช้ 4 ชุด):
  - Gold Futures (GC)
  - Crude Oil WTI (CL)
  - US Dollar Index Futures (DXY)
  - S&P500 (US500)
- ข้อมูลมีความไม่สมบูรณ์ (missing dates/values) หลัง merge จึงต้องทำ data preparation
""")

DATA_FILE = "gold_ml_dataset.csv"
try:
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"]).sort_values("Date")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Start", df["Date"].min().date().isoformat())
    c3.metric("End", df["Date"].max().date().isoformat())
    c4.metric("UP rate", f"{df['target'].mean():.3f}")

    with st.expander("Show dataset sample"):
        st.dataframe(df.head(12), use_container_width=True)
except Exception as e:
    st.warning(f"ไม่พบ `{DATA_FILE}` หรืออ่านไม่ได้: {e}")

st.divider()

# ---------- Features ----------
st.subheader("2) Features & Target")
st.markdown("""
**Features (4 ตัว):**
- `gold_ret` = daily return ของ gold close  
- `oil_ret`  = daily return ของ oil close  
- `dxy_ret`  = daily return ของ DXY  
- `sp_ret`   = daily return ของ S&P500  

**Return (pct_change):**  
`ret_t = (price_t - price_{t-1}) / price_{t-1}`

**Target:**  
`target = 1` ถ้า `gold_{t+1} > gold_t` (พรุ่งนี้ขึ้น) ไม่งั้น `0`
""")

st.divider()

# ---------- Data preparation ----------
st.subheader("3) Data Preparation (ทำไมข้อมูลไม่สมบูรณ์?)")
st.markdown("""
**สาเหตุความไม่สมบูรณ์:** ตลาดแต่ละตัวมีวันหยุด/วันเทรดไม่ตรงกัน ทำให้ merge ตาม Date แล้วเกิดช่องว่าง

**Pipeline:**
1. โหลดข้อมูลทั้ง 4 ชุด
2. แปลง `Date` ให้เป็นรูปแบบเดียวกัน แล้ว sort
3. merge ตาม `Date`
4. handle missing:
   - forward fill (ffill) สำหรับ series ของตลาด
   - dropna ที่เหลือ
5. สร้าง `target` จากราคาทองวันถัดไป
6. สร้าง features returns (`pct_change`)
7. split แบบ time-series (80/20, ไม่ shuffle)
8. scaling ด้วย `StandardScaler` (fit เฉพาะ train)
""")

st.divider()

# ---------- ML algorithm ----------
st.subheader("4) Machine Learning Algorithm (Ensemble)")
st.markdown("""
ใช้ **Voting Ensemble (Soft Voting)** รวมอย่างน้อย 3 โมเดล เช่น:
- Logistic Regression (linear classifier)
- Random Forest (bagging + decision trees)
- Gradient Boosting (boosting)

**Soft voting:** เอา probability ของแต่ละโมเดลมาถัวเฉลี่ย → ทำนาย class จาก threshold 0.5
""")

st.divider()

# ---------- Metrics (your real numbers) ----------
st.subheader("5) Evaluation (Ensemble)")
st.markdown("""
ค่าที่ได้จากการทดสอบ (Test set):
""")

ens_metrics = {
    "Accuracy": 0.5540540540540541,
    "Precision": 0.6106870229007634,
    "Recall": 0.625,
    "F1": 0.6177606177606177,
    "ROC-AUC": 0.5687333776595745,
}
cm = [[43, 51],
      [48, 80]]

colA, colB, colC, colD, colE = st.columns(5)
colA.metric("Accuracy", f"{ens_metrics['Accuracy']:.3f}")
colB.metric("Precision", f"{ens_metrics['Precision']:.3f}")
colC.metric("Recall", f"{ens_metrics['Recall']:.3f}")
colD.metric("F1", f"{ens_metrics['F1']:.3f}")
colE.metric("ROC-AUC", f"{ens_metrics['ROC-AUC']:.3f}")

st.markdown("**Confusion Matrix (Actual rows x Pred columns)**")
cm_df = pd.DataFrame(cm, columns=["Pred DOWN", "Pred UP"], index=["Actual DOWN", "Actual UP"])
st.table(cm_df)

st.divider()

st.subheader("6) References")
st.markdown("""
- Investing.com Historical Data (Gold, Oil, DXY, S&P500)
- scikit-learn: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
""")