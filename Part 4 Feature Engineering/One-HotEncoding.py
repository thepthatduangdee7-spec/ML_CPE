import pandas as pd

# โหลดข้อมูลตัวอย่าง
# (สมมติว่ามีคอลัมน์ชื่อ 'Color' เก็บค่า 'Red', 'Blue', 'Green')
data = {'User': ['A', 'B', 'C', 'D'],
        'Color': ['Red', 'Blue', 'Green', 'Red']}
df = pd.DataFrame(data)

print("--- Before One-Hot Encoding ---")
print(df)

# ใช้ฟังก์ชัน get_dummies ของ Pandas ในการทำ One-Hot Encoding
# dtype=int ใส่เพื่อให้ผลลัพธ์แสดงเป็นเลข 0 กับ 1 ชัดเจน (ไม่ใช่ True/False)
df_encoded = pd.get_dummies(df, columns=['Color'], dtype=int)

print("\n--- After One-Hot Encoding ---")
print(df_encoded)