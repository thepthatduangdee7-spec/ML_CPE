import pandas as pd

# โหลดไฟล์ข้อมูลต่อเนื่องจากการแก้ Missing Value มาทำต่อ
df = pd.read_csv("data_cleaned.csv")

print("--- [Compare] Rows Before Duplicate Removal ---")
print("Total rows:", len(df))
print("Duplicate count:", df.duplicated().sum())

# ลบข้อมูลที่ซ้ำกัน (keep='first' คือเก็บแถวแรกไว้)
df.drop_duplicates(inplace=True)

print("\n--- [Compare] Rows After Duplicate Removal ---")
print("Total rows:", len(df))
print("Duplicate count:", df.duplicated().sum())

# เซฟไฟล์ผลลัพธ์
df.to_csv("data_final_cleaned.csv", index=False)
print("\nSaved deduped dataset to 'data_final_cleaned.csv'")