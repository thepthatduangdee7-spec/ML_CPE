import pandas as pd

# โหลดไฟล์ล่าสุดมาทำความสะอาดต่อ
df = pd.read_csv("data_final_cleaned.csv")

print("--- [Compare] Maximum Duration Before Correction ---")
print(df['Duration'].describe())

# สมมติว่าเจอข้อมูลในคอลัมน์ 'Duration' แถวไหนที่มากกว่า 120 นาที 
# ให้แก้กลับมาเป็นค่าที่เหมาะสม เช่น 60 นาที (หรือระบุเงื่อนไขตามที่อาจารย์กำหนด)
df.loc[df['Duration'] > 120, 'Duration'] = 60

print("\n--- [Compare] Maximum Duration After Correction ---")
print(df['Duration'].describe())

# เซฟไฟล์ผลลัพธ์
df.to_csv("data_corrected.csv", index=False)
print("\nSaved corrected dataset to 'data_corrected.csv'")