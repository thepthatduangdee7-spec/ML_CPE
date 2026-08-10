import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Import ฟังก์ชันดึงข้อมูล
from dataset import load_utkface

# --- 1. โหลดข้อมูล ---
DATASET_PATH = r"C:\Users\uSeR\Desktop\UTKFace"
X_features, y_age, _ = load_utkface(DATASET_PATH, num_samples=2000)

# --- 2. แบ่งข้อมูล Train / Test ---
X_train, X_test, y_train_age, y_test_age = train_test_split(
    X_features, y_age, test_size=0.2, random_state=42
)

print("\n==========================================")
print("             LAB 1: REGRESSION            ")
print("==========================================")

# --- 3. Simple Linear Regression ---
center_pixel_idx = 512  # พิกเซลบริเวณกลางใบหน้า
X_train_simple = X_train[:, [center_pixel_idx]]
X_test_simple = X_test[:, [center_pixel_idx]]

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train_age)
y_pred_simple = model_simple.predict(X_test_simple)

print("\n[1] Simple Linear Regression (1 Pixel):")
print(f"  • MAE  : {mean_absolute_error(y_test_age, y_pred_simple):.2f} ปี")
print(f"  • R²   : {r2_score(y_test_age, y_pred_simple):.4f}")

# --- 4. Multiple Linear Regression ---
model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train_age)
y_pred_multiple = model_multiple.predict(X_test)

print("\n[2] Multiple Linear Regression (All Pixels):")
print(f"  • MAE  : {mean_absolute_error(y_test_age, y_pred_multiple):.2f} ปี")
print(f"  • R²   : {r2_score(y_test_age, y_pred_multiple):.4f}")

# --- 5. ตัวอย่างผลการทำนาย (Age Prediction S) ---
print("\n[3] Age Prediction Sample (5 ตัวอย่างแรก):")
for i in range(5):
    print(f"  • คนที่ {i+1}: อายุจริง = {y_test_age[i]} ปี | ทำนายได้ = {y_pred_multiple[i]:.1f} ปี")

print("==========================================\n")

# --- 6. วาดกราฟแสดงผล ---
plt.figure(figsize=(12, 5))

# กราฟ Simple Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(X_test_simple, y_test_age, color='blue', alpha=0.3, label='Actual Age')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Regression Line')
plt.title("Simple Linear Regression (Single Pixel vs Age)")
plt.xlabel("Pixel Intensity (Scaled)")
plt.ylabel("Age")
plt.legend()

# กราฟ Multiple Linear Regression
plt.subplot(1, 2, 2)
plt.scatter(y_test_age, y_pred_multiple, color='green', alpha=0.4)
plt.plot([y_test_age.min(), y_test_age.max()], [y_test_age.min(), y_test_age.max()], 'r--', lw=2)
plt.title("Multiple Linear Regression (All Pixels): Actual vs Predicted")
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")

plt.tight_layout()
plt.show()