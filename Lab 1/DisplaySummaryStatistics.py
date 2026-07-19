import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

# Display summary statistics
print("Summary Statistics:")
print(df.describe())