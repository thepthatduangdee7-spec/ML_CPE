import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

# *** ให้เปลี่ยนคำว่า 'Duration' เป็นชื่อคอลัมน์ที่เป็น Class ของคุณจริงๆ ***
class_column = 'Duration' 

print(f"--- Class Distribution for [{class_column}] ---")
# 1. นับจำนวนแยกตามกลุ่ม
print(df[class_column].value_counts())

print("\n--- Percentage (%) ---")
# 2. คิดออกมาเป็นเปอร์เซ็นต์
print(df[class_column].value_counts(normalize=True) * 100)