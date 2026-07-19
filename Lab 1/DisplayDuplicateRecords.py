import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

print("--- Duplicate Records Analysis ---")
# 1. นับจำนวนแถวที่ซ้ำทั้งหมด
duplicate_count = df.duplicated().sum()
print(f"Total duplicate rows: {duplicate_count}")

# 2. ถ้ามีข้อมูลซ้ำ ให้แสดงแถวที่ซ้ำออกมาดู
if duplicate_count > 0:
    print("\n--- Examples of Duplicate Rows ---")
    print(df[df.duplicated()])