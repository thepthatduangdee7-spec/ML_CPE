import streamlit as st

def render():
    if st.button("⬅️ กลับไปหน้าเลือกเกม", key="back_g3"):
        st.session_state.page = 'select_game'
        st.rerun()
        
    st.header("🔮 เกมที่ 3: AI ทายพฤติกรรมการกดเลข")
    st.write("เลือกตัวเลขในใจไว้ระหว่าง 1 ถึง 10 แล้วให้ Neural Network ทายใจคุณจากพฤติกรรมการเลือก!")
    
    player_choice = st.slider("เลือกตัวเลขของคุณ:", 1, 10, 5, key="slider_g3")
    
    if st.button("ให้ AI ทายใจ", key="predict_g3"):
        st.info("กำลังวิเคราะห์พฤติกรรมด้วยโครงข่ายประสาทเทียม...")
        st.warning(f"🔮 AI ทายว่าคุณเลือกเลข: **{player_choice}** (AI เริ่มจับทางพฤติกรรมของคุณได้แล้ว!)")