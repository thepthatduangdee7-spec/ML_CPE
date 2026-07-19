import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

print("--- Mean of each column ---")
# คำนวณหาค่าเฉลี่ยเฉพาะคอลัมน์ที่เป็นตัวเลข
print(df.mean(numeric_only=True))