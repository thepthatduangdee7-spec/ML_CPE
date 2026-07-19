import pandas as pd
from sklearn.preprocessing import LabelEncoder

# โหลดข้อมูลตัวอย่าง
# (สมมติว่าใน dataset มีคอลัมน์ชื่อ 'Level' ที่เก็บค่า 'Low', 'Medium', 'High')
# เพื่อให้เห็นภาพชัดเจน จะขอสร้าง Data สมมติขึ้นมาจำลองให้ดูครับ
data = {'User': ['A', 'B', 'C', 'D'],
        'Level': ['Low', 'High', 'Medium', 'Low']}
df = pd.DataFrame(data)

print("--- Before Label Encoding ---")
print(df)

# เรียกใช้ LabelEncoder
encoder = LabelEncoder()

# ทำการแปลงข้อความในคอลัมน์ 'Level' ให้เป็นตัวเลข
df['Level_Encoded'] = encoder.fit_transform(df['Level'])

print("\n--- After Label Encoding ---")
print(df)

# แสดงตารางคู่มือบอกว่าคำไหนถูกเปลี่ยนเป็นเลขอะไร
mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("\nMapping Rules:", mapping)