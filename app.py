import streamlit as st

st.set_page_config(page_title="Gold Prediction Project", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
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
</style>
""", unsafe_allow_html=True)

st.title("Gold Price Direction Prediction (Project)")

st.markdown("""
<div class="card">
  <div style="font-size:1.25rem;font-weight:900;letter-spacing:-0.02em;">Project Overview</div>
  <div class="small-muted" style="margin-top:8px;">
    โปรเจกต์นี้ใช้ dataset อย่างน้อย 2 ชุด (จริง ๆ ใช้ 4 ชุด: Gold, Oil, DXY, S&amp;P500)
    และสร้างโมเดล 2 ประเภท:
    <ul>
      <li><b>Machine Learning Ensemble</b> (รวม 3+ โมเดลด้วย Voting)</li>
      <li><b>Neural Network</b> (MLP ออกแบบเอง)</li>
    </ul>
    เมนูด้านซ้ายจะมี 4 หน้า (ตรงตาม requirement):
    <ul>
      <li>อธิบาย ML</li>
      <li>อธิบาย NN</li>
      <li>ทดสอบ ML</li>
      <li>ทดสอบ NN</li>
    </ul>
  </div>
</div>
""", unsafe_allow_html=True)

st.info("เปิด Sidebar (ซ้าย) แล้วเลือกหน้า: ML Explain / NN Explain / ML Test / NN Test")