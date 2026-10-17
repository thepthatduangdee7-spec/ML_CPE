import streamlit as st
import numpy as np
import random
from PIL import Image

def render(model_cnn):
    if st.button("⬅️ กลับไปหน้าเลือกเกม", key="back_g2"):
        st.session_state.page = 'select_game'
        st.rerun()
        
    st.header("👗 เกมที่ 2: ภารกิจล่าไอเทมแฟชั่น (DCNN)")
    st.write("โจทย์: AI จะสุ่มภารกิจให้คุณหาไอเทมมาส่ง โดยใช้ **กล้องถ่ายรูป (Webcam บนคอม / กล้องมือถือ)** ถ่ายภาพสิ่งของจริงรอบตัวส่งให้ AI ตรวจสอบ!")

    fashion_labels = [
        "เสื้อยืด/เสื้อผ้า (T-shirt/top)", 
        "กางเกงขายาว (Trouser)", 
        "เสื้อกันหนาว (Pullover)", 
        "ชุดเดรส (Dress)", 
        "เสื้อโค้ท (Coat)", 
        "รองเท้าแตะ (Sandal)", 
        "เสื้อเชิ้ต (Shirt)", 
        "รองเท้าผ้าใบ (Sneaker)", 
        "กระเป๋าถือ (Bag)", 
        "รองเท้าบูท (Ankle boot)"
    ]

    if 'target_class_idx' not in st.session_state:
        st.session_state.target_class_idx = random.randint(0, 9)

    target_name = fashion_labels[st.session_state.target_class_idx]

    st.markdown(f"### 🎯 **ภารกิจของคุณ:** กรุณาใช้กล้องถ่ายภาพ **'{target_name}'** มาส่งให้ AI!")

    if st.button("🎲 เปลี่ยนโจทย์ใหม่", key="change_target"):
        st.session_state.target_class_idx = random.randint(0, 9)
        st.rerun()

    st.markdown("---")

    camera_file = st.camera_input("📸 ถ่ายรูปไอเทมของคุณด้วยกล้อง")

    input_image = None
    if camera_file is not None:
        image = Image.open(camera_file)
        img_gray = image.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
        input_image = np.array(img_gray).astype('float32') / 255.0

    st.markdown("---")
    
    if st.button("🔍 ส่งตรวจสอบกับ DCNN", key="predict_g2"):
        if model_cnn is not None and input_image is not None:
            X_cnn = input_image.reshape(1, 28, 28, 1)
            
            preds = model_cnn.predict(X_cnn)
            pred_class = np.argmax(preds)
            confidence = np.max(preds) * 100
            
            detected_name = fashion_labels[pred_class]
            
            st.info(f"🤖 AI วิเคราะห์ภาพจากกล้องว่าเป็น: **{detected_name}** (ความมั่นใจ {confidence:.2f}%)")
            
            if pred_class == st.session_state.target_class_idx:
                st.success(f"🎉 **Mission Complete!** ยอดเยี่ยมมาก คุณทำภารกิจสำเร็จแล้ว!")
                st.session_state.target_class_idx = random.randint(0, 9)
            else:
                st.warning(f"❌ ยังไม่ถูกต้อง AI มองว่าเป็น '{detected_name}' ลองถ่ายใหม่อีกครั้งนะ!")
        else:
            st.warning("กรุณาเปิดกล้องและถ่ายรูปไอเทมก่อนกดตรวจสอบครับ!")