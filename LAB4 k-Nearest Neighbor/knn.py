import pandas as pd

# โหลดชุดข้อมูล penguins_raw.csv
df = pd.read_csv(r"C:\Users\useR\Desktop\LAB4 k-Nearest Neighbor\penguins_raw.csv")

# จัดการข้อมูลที่เป็นค่าว่าง (NaN) เพราะข้อมูลดิบมักจะมีค่าหายไป
df = df.dropna()

# เลือกคอลัมน์ฟีเจอร์ที่เป็นตัวเลข และคอลัมน์เป้าหมาย (Species)
feature_cols = [
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
]
X = df[feature_cols]
y = df["Species"]

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_values = [3, 5, 7]

print("\n" + "=" * 45)
print("       KNN CLASSIFICATION RESULTS")
print("=" * 45)

results = []

for k in k_values:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  results.append((k, accuracy))
  print(f"  k = {k:<3} | Accuracy = {accuracy*100:.2f}%")

print("=" * 45)

best_k = max(results, key=lambda x: x[1])

print("\n🏆 BEST MODEL")
print("-" * 45)
print(f"Best k value     : {best_k[0]}")
print(f"Best Accuracy    : {best_k[1]*100:.2f}%")
print("-" * 45)
print("Dataset          : Palmer Penguins (Raw)")
print("Algorithm        : K-Nearest Neighbor (KNN)")
print(f"Features         : {len(feature_cols)} attributes")
print("Test Size        : 20%")
print("=" * 45)

print("\nConclusion:")
print(f"The best k value is {best_k[0]} with accuracy {best_k[1]*100:.2f}%")