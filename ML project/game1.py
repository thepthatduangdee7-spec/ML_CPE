import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def render(model_mlp, scaler_mlp):
    if st.button("⬅️ กลับไปหน้าเลือกเกม", key="back_g1"):
        st.session_state.page = 'select_game'
        st.rerun()
        
    st.header("✏️ เกมที่ 1: วาดตัวเลขในใจ (0-9)")
    st.write("วาดตัวเลขลงในกรอบ แล้วให้ AI ทาย หากทายผิด สามารถเลือกคำตอบที่ถูกต้องเพื่อบันทึกสะสมลง CSV ได้!")
    
    if st.button("🔄 วาดใหม่ (Clear Canvas)", key="clear_g1"):
        st.session_state.canvas_key += 1
        if 'last_pred' in st.session_state:
            del st.session_state.last_pred
        st.rerun()

    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=150,
        width=150,
        drawing_mode="freedraw",
        return_image_data=True,
        key=f"canvas_num_{st.session_state.canvas_key}",
    )
    
    if st.button("ให้ AI ทายตัวเลขนี้", key="predict_g1"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            img_gray = img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
            img_arr = np.array(img_gray)
            
            X_pred = (img_arr.flatten().reshape(1, -1) / 255.0) * 16.0
            X_pred_scaled = scaler_mlp.transform(X_pred)
            
            st.session_state.last_X_pred = X_pred[0]
            
            prediction = model_mlp.predict(X_pred_scaled)[0]
            probabilities = model_mlp.predict_proba(X_pred_scaled)[0]
            confidence = probabilities[prediction] * 100
            
            st.session_state.last_pred = prediction
            st.success(f"🤖 AI ทายว่าคุณวาดเลข: **{prediction}** (ความมั่นใจ {confidence:.2f}%)")
        else:
            st.warning("กรุณาวาดตัวเลขลงในกรอบก่อนกดทายครับ!")

    if 'last_pred' in st.session_state:
        st.markdown("---")
        st.subheader("🛠️ สอน AI เพิ่มเติม (บันทึกข้อมูลลง CSV)")
        st.write(f"หาก AI ทายผิด (AI ทายว่า {st.session_state.last_pred}) สามารถเลือกตัวเลขที่ถูกต้องแล้วบันทึกเก็บไว้ได้:")
        
        true_label = st.selectbox("ตัวเลขที่ถูกต้องจริงๆ คือ:", list(range(10)), key="true_label_select")
        
        if st.button("💾 บันทึกคำตอบที่ถูกต้องลง CSV", key="save_csv"):
            if 'last_X_pred' in st.session_state:
                csv_file = "feedback_data.csv"
                pixel_data = st.session_state.last_X_pred
                
                row_data = {f"pixel_{i}": val for i, val in enumerate(pixel_data)}
                row_data['label'] = true_label
                
                df_new = pd.DataFrame([row_data])
                
                if os.path.exists(csv_file):
                    df_new.to_csv(csv_file, mode='a', header=False, index=False)
                else:
                    df_new.to_csv(csv_file, index=False)
                
                st.success("🎉 บันทึกข้อมูลลงไฟล์ `feedback_data.csv` เรียบร้อยแล้ว!")
                del st.session_state.last_pred