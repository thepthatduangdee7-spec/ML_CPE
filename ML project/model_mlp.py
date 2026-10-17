import pickle
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def train_and_save_model(new_X=None, new_y=None):
    digits = load_digits()
    X, y = digits.data, digits.target

    # ถ้ารับข้อมูลใหม่ที่ผู้เล่นสอนเพิ่มเข้ามา นำมาต่อรวมกับชุดข้อมูลหลัก
    if new_X is not None and new_y is not None and len(new_X) > 0:
        X = np.vstack([X, np.array(new_X)])
        y = np.concatenate([y, np.array(new_y)])
        print(f"กำลังอัปเดตโมเดลด้วยข้อมูลใหม่รวมทั้งหมด {len(new_y)} ตัว...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        max_iter=300, 
        random_state=42,
        early_stopping=True
    )
    mlp_model.fit(X_scaled, y)

    # บันทึกไฟล์ทับของเดิม
    with open("model_mlp.pkl", "wb") as f:
        pickle.dump((mlp_model, scaler), f)
    print("อัปเดตและบันทึกไฟล์โมเดลเรียบร้อยแล้ว!")

if __name__ == "__main__":
    train_and_save_model()