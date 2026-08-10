import os, glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ==========================================
# 0. ฟังก์ชันดึงรูปภาพ UTKFace (แบบย่อ)
# ==========================================
def load_utkface(path, num_samples=2000):
    img_paths = glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True)[:num_samples]
    X, y = [], []
    for p in img_paths:
        parts = os.path.basename(p).split('_')
        if len(parts) >= 3 and parts[1].isdigit():
            img = Image.open(p).convert('L').resize((32, 32))
            X.append(np.array(img).flatten() / 255.0)
            y.append(int(parts[1]))  # 0 = Male, 1 = Female
    return np.array(X), np.array(y)

# ⚠️ เปลี่ยน Path ให้ตรงกับโฟลเดอร์รูปภาพบนเครื่องของคุณ
DATASET_PATH = r"C:\Users\uSeR\Desktop\UTKFace"
X_features, y_gender = load_utkface(DATASET_PATH, 2000)


# ==========================================
# 1. Preparing Classification Data
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_gender, test_size=0.2, random_state=42, stratify=y_gender
)

# Scaling + PCA ลดมิติเหลือ 2D
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

print("=== 1) Preparing Classification Data ===")
print(f"Total: {len(X_features)} | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"PCA Dimension: {X_features.shape[1]} -> {X_train_pca.shape[1]} features\n")


# ==========================================
# 2. Logistic Regression
# ==========================================
model = LogisticRegression(random_state=42)
model.fit(X_train_pca, y_train)

print("=== 2) Logistic Regression ===")
print(f"Train Accuracy : {model.score(X_train_pca, y_train)*100:.2f}%")
print(f"Test Accuracy  : {model.score(X_test_pca, y_test)*100:.2f}%\n")


# ==========================================
# 3. Gender Prediction
# ==========================================
y_pred = model.predict(X_test_pca)
gender_map = {0: 'Male', 1: 'Female'}

print("=== 3) Gender Prediction (Sample 5 Cases) ===")
for i in range(5):
    act, prd = gender_map[y_test[i]], gender_map[y_pred[i]]
    result = "Correct ✅" if act == prd else "Wrong ❌"
    print(f"Sample {i+1}: Actual = {act:<6} | Predicted = {prd:<6} -> {result}")
print()


# ==========================================
# 4. Confusion Matrix
# ==========================================
cm = confusion_matrix(y_test, y_pred)
print("=== 4) Confusion Matrix ===")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Male', 'Female']))


# ==========================================
# 5. Decision Boundary Visualization
# ==========================================
print("=== 5) Decision Boundary Visualization (Displaying Plot...) ===")

plt.figure(figsize=(10, 4.5))

# Plot 1: Decision Boundary
plt.subplot(1, 2, 1)
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', alpha=0.7)
plt.title("Decision Boundary (PCA 2D)")

# Plot 2: Confusion Matrix Display
plt.subplot(1, 2, 2)
ConfusionMatrixDisplay(cm, display_labels=['Male', 'Female']).plot(ax=plt.gca(), cmap='Blues')
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()