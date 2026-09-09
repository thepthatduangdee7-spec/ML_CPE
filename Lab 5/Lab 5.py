import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train SVM with 3 Kernels (Linear, Polynomial, RBF)
kernels = ['linear', 'poly', 'rbf']
accuracies = {}
predictions = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    y_pred = svm.predict(X_test_scaled)
    predictions[kernel] = y_pred
    accuracies[kernel] = accuracy_score(y_test, y_pred) * 100

# ==============================================================================
# OUTPUT PRINTING
# ==============================================================================
print("="*55)
print("1. ACCURACY SCORES FOR EACH SVM KERNEL")
print("="*55)
for kernel, acc in accuracies.items():
    print(f"Kernel: {kernel.capitalize():<10} | Accuracy: {acc:.2f}%")

print("\n" + "="*55)
print("2. PREDICTIONS ON DATASET (Sample: First 15 Samples)")
print("="*55)
print("Actual Target Labels: ", list(y_test[:15]))
print("RBF Predicted Labels: ", list(predictions['rbf'][:15]))
print("="*55)

# ==============================================================================
# GRAPH OUTPUT
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# กราฟที่ 1: เปรียบเทียบ Accuracy Score แต่ละ Kernel
kernel_names = [k.capitalize() for k in accuracies.keys()]
acc_values = list(accuracies.values())
bars = axes[0].bar(kernel_names, acc_values, color=['#4C72B0', '#55A868', '#C44E52'])

axes[0].set_title("SVM Kernel Accuracy Comparison", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Accuracy Score (%)", fontsize=12)
axes[0].set_ylim(0, 110)

for bar in bars:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

axes[0].grid(axis='y', linestyle='--', alpha=0.6)

# กราฟที่ 2: เปรียบเทียบ Actual vs Predicted (15 ตัวอย่างแรกของ RBF Kernel)
sample_size = 15
x_axis = np.arange(sample_size)
axes[1].scatter(x_axis, y_test[:15], color='blue', label='Actual Label', s=100, marker='o')
axes[1].scatter(x_axis, predictions['rbf'][:15], color='red', label='Predicted Label (RBF)', s=40, marker='x')

axes[1].set_title("Actual vs Predicted Labels (First 15 Samples)", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Sample Index", fontsize=12)
axes[1].set_ylabel("Class Label (0, 1, 2)", fontsize=12)
axes[1].set_yticks([0, 1, 2])
axes[1].set_yticklabels(iris.target_names)
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()