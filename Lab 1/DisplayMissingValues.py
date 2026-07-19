import pandas as pd

# 1. Load dataset (โหลดข้อมูล)
df = pd.read_csv("data.csv")

# 2. Display Missing Values
print("--- Missing Values per Column ---")
print(df.isnull().sum())