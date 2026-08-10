import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)

# Import ฟังก์ชันดึงข้อมูล
from dataset import load_utkface

# --- 1. โหลดข้อมูล ---
DATASET_PATH = r"C:\Users\uSeR\Desktop\UTKFace"
X_features, y_age, y_gender = load_utkface(DATASET_PATH, num_samples=2000)

# --- 2. การเปรียบเทียบ REGRESSION (ทำนายอายุ) ---
X_tr_r, X_te_r, y_tr_age, y_te_age = train_test_split(X_features, y_age, test_size=0.2, random_state=42)

# Simple vs Multiple
m_simple = LinearRegression().fit(X_tr_r[:, [512]], y_tr_age)
y_p_simple = m_simple.predict(X_te_r[:, [512]])

m_multi = LinearRegression().fit(X_tr_r, y_tr_age)
y_p_multi = m_multi.predict(X_te_r)

print("==========================================")
print("1. REGRESSION METRICS COMPARISON (Age)")
print("==========================================")
df_reg = pd.DataFrame({
    'Metric': ['MAE (ปี)', 'RMSE (ปี)', 'R-squared'],
    'Simple Linear': [
        mean_absolute_error(y_te_age, y_p_simple),
        np.sqrt(mean_squared_error(y_te_age, y_p_simple)),
        r2_score(y_te_age, y_p_simple)
    ],
    'Multiple Linear': [
        mean_absolute_error(y_te_age, y_p_multi),
        np.sqrt(mean_squared_error(y_te_age, y_p_multi)),
        r2_score(y_te_age, y_p_multi)
    ]
})
print(df_reg.to_string(index=False))

print("\n------------------------------------------")
print("TRAINING VS TESTING PERFORMANCE (Check Overfitting)")
print(f"Train R^2: {m_multi.score(X_tr_r, y_tr_age):.4f}")
print(f"Test R^2 : {m_multi.score(X_te_r, y_te_age):.4f}")

# --- 3. การเปรียบเทียบ CLASSIFICATION (จำแนกเพศ) ---
X_tr_c, X_te_c, y_tr_gen, y_te_gen = train_test_split(X_features, y_gender, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_c)
X_te_sc = scaler.transform(X_te_c)

pca = PCA(n_components=50) # ใช้ 50 PCA components
X_tr_pca = pca.fit_transform(X_tr_sc)
X_te_pca = pca.transform(X_te_sc)

m_cls = LogisticRegression().fit(X_tr_pca, y_tr_gen)
y_p_gen = m_cls.predict(X_te_pca)
y_prob_gen = m_cls.predict_proba(X_te_pca)[:, 1]

print("\n==========================================")
print("2. CLASSIFICATION METRICS (Gender)")
print("==========================================")
print(f"Accuracy  : {accuracy_score(y_te_gen, y_p_gen):.4f}")
print(f"Precision : {precision_score(y_te_gen, y_p_gen):.4f}")
print(f"Recall    : {recall_score(y_te_gen, y_p_gen):.4f}")
print(f"F1-score  : {f1_score(y_te_gen, y_p_gen):.4f}")

# --- 4. วาด ROC Curve & AUC ---
fpr, tpr, _ = roc_curve(y_te_gen, y_prob_gen)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gender Classification')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()