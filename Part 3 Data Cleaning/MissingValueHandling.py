import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

print("--- [Compare] Missing Values Before ---")
print(df.isnull().sum())

# คำนวณหาค่า Median ของคอลัมน์ Calories
calories_median = df['Calories'].median()

# แทนที่ช่องว่าง (Missing Value) ด้วยค่า Median
df['Calories'] = df['Calories'].fillna(calories_median)

print("\n--- [Compare] Missing Values After Handling ---")
print(df.isnull().sum())

# เซฟไฟล์ผลลัพธ์ออกไปใช้งานต่อ
df.to_csv("data_cleaned.csv", index=False)
print("\nSaved handled dataset to 'data_cleaned.csv'")