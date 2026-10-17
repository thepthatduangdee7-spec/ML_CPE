import streamlit as st
import pickle
import tensorflow as tf

# นำเข้ามินิเกมทั้ง 3 ไฟล์ที่แยกไว้
import game1
import game2
import game3

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Mini-Game Arcade", page_icon="🎮", layout="centered")

# --- ตกแต่งหน้าตาด้วย CSS Custom Style ---
st.markdown("""
    <style>
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        color: #ffffff;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- จัดการสถานะหน้า (Session State) ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

# --- โหลดโมเดล MLP และ Scaler ---
@st.cache_resource
def load_mlp_model():
    with open("model_mlp.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    return model, scaler

try:
    model_mlp, scaler_mlp = load_mlp_model()
except Exception as e:
    st.error(f"ไม่พบไฟล์ 'model_mlp.pkl' กรุณารันไฟล์ model_mlp.py ก่อนครับ: {e}")

# --- โหลดโมเดล DCNN ---
@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model("model_cnn.h5")
        return model
    except:
        return None

model_cnn = load_cnn_model()

# ====================================================
# 1. หน้าแรก (Homepage) - หน้าหลักของแอป
# ====================================================
if st.session_state.page == 'home':
    st.title("🎮 AI Mini-Game Arcade")
    st.write("ยินดีต้อนรับสู่แอปพลิเคชันรวมมินิเกมที่ขับเคลื่อนด้วยโครงข่ายประสาทเทียม (Neural Networks) และ Machine Learning!")
    
    st.markdown("---")
    
    if st.button("🚀 เริ่ม (Start)", use_container_width=True, type="primary"):
        st.session_state.page = 'select_game'
        st.rerun()
        
    if st.button("📖 วิธีเล่น (How to Play)", use_container_width=True):
        st.info("""
        **📌 คำแนะนำในการเล่น:**
        - กดปุ่ม **'เริ่ม'** เพื่อเข้าสู่หน้าเลือกมินิเกม
        - มีเกมให้เลือกเล่นทั้ง 3 เกม (วาดตัวเลข MLP, ภารกิจล่าไอเทม DCNN, และทายพฤติกรรม)
        """)

# ====================================================
# 2. หน้าเลือกเกม (Game Selection Menu)
# ====================================================
elif st.session_state.page == 'select_game':
    st.title("🎯 เลือกมินิเกมที่คุณต้องการเล่น")
    st.write("กรุณาเลือกมินิเกมด้านล่างนี้ได้เลยครับ:")
    
    st.markdown("---")
    
    if st.button("✏️ เกมที่ 1: วาดตัวเลขในใจ (MLP)", use_container_width=True):
        st.session_state.page = 'game1'
        st.rerun()
        
    if st.button("🎨 เกมที่ 2: ภารกิจล่าไอเทมแฟชั่น (DCNN)", use_container_width=True):
        st.session_state.page = 'game2'
        st.rerun()
        
    if st.button("🔮 เกมที่ 3: ทายพฤติกรรมการกดเลข", use_container_width=True):
        st.session_state.page = 'game3'
        st.rerun()
        
    st.markdown("---")
    if st.button("⬅️ กลับหน้าแรก", use_container_width=True):
        st.session_state.page = 'home'
        st.rerun()

# ====================================================
# 3. เรียกใช้งานเกมจากไฟล์แยก
# ====================================================
elif st.session_state.page == 'game1':
    game1.render(model_mlp, scaler_mlp)

elif st.session_state.page == 'game2':
    game2.render(model_cnn)

elif st.session_state.page == 'game3':
    game3.render()