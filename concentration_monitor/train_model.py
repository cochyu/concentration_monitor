import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# ── 读取数据 ──────────────────────────────────
X, y = [], []
with open("data/training_data.csv") as f:
    for row in csv.DictReader(f):
        X.append([float(row["ear"]), float(row["mar"]),
                  float(row["pitch"]), float(row["yaw"])])
        y.append(int(row["label"]))

X = np.array(X)
y = np.array(y)
print(f"数据加载完成：共 {len(y)} 条")

# ── 划分训练集/测试集 ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集: {len(X_train)}条  测试集: {len(X_test)}条")

# ── 特征标准化 ────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 训练SVM ───────────────────────────────────
print("正在训练SVM模型...")
model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
model.fit(X_train, y_train)

# ── 评估 ──────────────────────────────────────
y_pred = model.predict(X_test)
acc = np.mean(y_pred == y_test) * 100
print(f"\n✅ 测试集准确率：{acc:.1f}%")
print("\n分类报告：")
print(classification_report(y_test, y_pred,
      target_names=["专注", "分心", "疲劳"]))
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

# ── 保存模型 ──────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model,  "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\n模型已保存到 models/ 文件夹")