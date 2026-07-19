import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("DATASET.py")

# 1. คำนวณหาค่าความสัมพันธ์ (Correlation) เฉพาะคอลัมน์ที่เป็นตัวเลข
corr_matrix = df.corr(numeric_only=True)

# 2. ตั้งขนาดของรูปกราฟ
plt.figure(figsize=(8, 6))

# 3. วาด Heatmap 
# annot=True คือให้แสดงตัวเลขค่าความสัมพันธ์ลงไปในบล็อกด้วย
# cmap='coolwarm' คือใช้โทนสีน้ำเงิน (ติดลบ) ถึงสีแดง (เป็นบวก)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# ใส่ชื่อกราฟ
plt.title('Correlation Heatmap', fontsize=14)

# แสดงรูปกราฟออกมา
plt.show()