import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 1. Load Dataset & Preprocess
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Setup Configurations & Epochs to Compare
configs = {
    "Config A (1 Layer: 10)": (10,),
    "Config B (2 Layers: 10, 10)": (10, 10),
    "Config C (2 Layers: 20, 10)": (20, 10)
}
epochs_list = [20, 50, 200]

results = []
models_history = {}

# 3. Train Models and Collect Data
for cfg_name, hidden_layers in configs.items():
    for ep in epochs_list:
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=ep,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            learning_rate_init=0.01
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        train_loss = model.loss_curve_[-1]
        val_acc = model.validation_scores_[-1]
        
        results.append({
            "Config": cfg_name,
            "Epochs": ep,
            "Test Accuracy (%)": round(acc * 100, 2),
            "Train Loss": round(train_loss, 4),
            "Val Accuracy (%)": round(val_acc * 100, 2),
            "Predictions": y_pred
        })
        
        models_history[f"{cfg_name} - {ep} ep"] = {
            "loss": model.loss_curve_,
            "val_score": model.validation_scores_
        }

# ==============================================================================
# OUTPUT PRINTING (ตรงตามข้อกำหนดทั้ง 4 ข้อ)
# ==============================================================================

print("="*65)
print("1. ACCURACY SCORES FOR EACH NN CONFIGURATION &")
print("2. COMPARISON OF ACCURACY USING DIFFERENT NUMBERS OF EPOCHS")
print("="*65)
df_results = pd.DataFrame(results)[["Config", "Epochs", "Test Accuracy (%)"]]
print(df_results.to_string(index=False))

print("\n" + "="*65)
print("3. TRAINING AND VALIDATION ACCURACY / LOSS RESULTS")
print("="*65)
df_details = pd.DataFrame(results)[["Config", "Epochs", "Train Loss", "Val Accuracy (%)", "Test Accuracy (%)"]]
print(df_details.to_string(index=False))

print("\n" + "="*65)
print("4. PREDICTIONS ON THE SELECTED DATASET (Sample: First 15 Samples)")
print("="*65)
best_run = results[-1] # ดึงผลลัพธ์รอบสุดท้ายมาแสดงตัวอย่าง
print("Actual Target Labels: ", list(y_test[:15]))
print("NN Predicted Labels:  ", list(best_run["Predictions"][:15]))
print("="*65)

# ==============================================================================
# GRAPH OUTPUT (สำหรับใส่รายงาน)
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# กราฟที่ 1: เปรียบเทียบ Accuracy ตามจำนวน Epochs ของแต่ละ Config
for cfg_name in configs.keys():
    sub_df = [r for r in results if r["Config"] == cfg_name]
    eps = [r["Epochs"] for r in sub_df]
    accs = [r["Test Accuracy (%)"] for r in sub_df]
    axes[0].plot(eps, accs, marker='o', label=cfg_name, linewidth=2)

axes[0].set_title("Accuracy vs. Epochs", fontweight='bold')
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Test Accuracy (%)")
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# กราฟที่ 2: Training Loss Results
for key, data in models_history.items():
    if "200 ep" in key: # สุ่มแสดงเฉพาะรุ่น 200 epochs เพื่อไม่ให้สายรกเกินไป
        axes[1].plot(data["loss"], label=key)

axes[1].set_title("Training Loss Curve (200 Epochs)", fontweight='bold')
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

# กราฟที่ 3: Validation Accuracy Results
for key, data in models_history.items():
    if "200 ep" in key:
        val_acc_pct = [v * 100 for v in data["val_score"]]
        axes[2].plot(val_acc_pct, label=key)

axes[2].set_title("Validation Accuracy Curve (200 Epochs)", fontweight='bold')
axes[2].set_xlabel("Epochs")
axes[2].set_ylabel("Validation Accuracy (%)")
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()