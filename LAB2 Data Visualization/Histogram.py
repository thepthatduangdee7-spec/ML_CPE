import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("DATASET.py")

# ตั้งค่าสไตล์ของกราฟให้ดูสวยงาม
sns.set_theme(style="whitegrid")

# สร้าง Histogram สำหรับคอลัมน์ที่ต้องการ (ตัวอย่างเช่น 'Calories')
# สามารถเปลี่ยน 'Calories' เป็นคอลัมน์อื่นได้ เช่น 'Duration', 'Pulse'
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Calories', kde=True, color='skyblue')

# ใส่ชื่อกราฟและชื่อแกน
plt.title('Histogram of Calories Distribution', fontsize=14)
plt.xlabel('Calories', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)

# แสดงรูปกราฟออกมา
plt.show()