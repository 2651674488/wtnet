import pandas as pd
import matplotlib.pyplot as plt

# 读取日志数据
df = pd.read_csv("../logs/ltf/exchange_/version_6/metrics.csv")

# 过滤无效值（如test_mse在验证阶段为NaN）
df = df.dropna(subset=["val_mse", "val_mae"])

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 子图1：MSE和MAE曲线
ax1.plot(df["epoch"], df["val_mse"], label="Val MSE", color="blue")
ax1.plot(df["epoch"], df["test_mse"], label="Test MSE", linestyle="--", color="blue")
ax1.plot(df["epoch"], df["val_mae"], label="Val MAE", color="red")
ax1.plot(df["epoch"], df["test_mae"], label="Test MAE", linestyle="--", color="red")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Metric Value")
ax1.set_title("Validation and Test Metrics")
ax1.legend()
ax1.grid(True)

# 子图2：学习率曲线
ax2.plot(df["epoch"], df["lr"], label="Learning Rate", color="green")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Learning Rate")
ax2.set_title("Learning Rate Schedule")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_metrics.png")  # 保存图片
plt.show()