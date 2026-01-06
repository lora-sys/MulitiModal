import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. 加载 0.6/0.8 风格的特征矩阵
df = pd.read_csv("D:/repos/mulitModal/iteration0.8/training_dataset.csv")
X = df.drop("label", axis=1)
y = df["label"]

# 2. 划分训练集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 训练 AI 大脑
print("🧠 AI 正在学习按摩模式...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. 评估 (打印你关心的那个报告)
y_pred = clf.predict(X_test)
print("\n🎯 模型考核结果:")
print(f"总准确率: {accuracy_score(y_test, y_pred):.2%}")
print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=["空载", "柔和", "深度"]))

# 5. 【新增】保存模型，以后直接给传感器数据就能预测了
model_path = "D:/repos/mulitModal/iteration0.8/massage_ai_model.pkl"
joblib.dump(clf, model_path)
print(f"\n💾 模型已保存至: {model_path}")
