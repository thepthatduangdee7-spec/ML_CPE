import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

print("--- Median of each column ---")
# คำนวณหาค่ากึ่งกลางเฉพาะคอลัมน์ที่เป็นตัวเลข
print(df.median(numeric_only=True))