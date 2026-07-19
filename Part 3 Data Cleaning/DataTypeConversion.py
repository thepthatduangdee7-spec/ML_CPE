import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data_corrected.csv")

# Display data types before conversion
print("Data Types Before Conversion:")
print(df.dtypes)


# Convert data types
df["Duration"] = df["Duration"].astype(float)
df["Pulse"] = df["Pulse"].astype(float)
df["Maxpulse"] = df["Maxpulse"].astype(float)


# Display data types after conversion
print("\nData Types After Conversion:")
print(df.dtypes)


# Save converted dataset
df.to_csv("data_type_converted.csv", index=False)

print("\nData type conversion completed.")