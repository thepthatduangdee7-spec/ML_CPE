import pandas as pd

# เปลี่ยน 'your_dataset.csv' เป็นชื่อไฟล์จริงของคุณ
df = pd.read_csv('data.csv')

# แสดง 5 แถวแรกเพื่อตรวจดูหน้าตาข้อมูล
print("--- 5 แถวแรกของข้อมูล ---")
print(df.head())