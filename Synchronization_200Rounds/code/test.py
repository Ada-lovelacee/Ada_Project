import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/metrics_On_Test.csv')

# 创建子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6), dpi=300)

# 绘制 Loss
ax1.plot(df["location"], df["loss"], marker='o', color='red')
ax1.set_xlabel("Scenario")
ax1.set_ylabel("loss")
ax1.set_title("loss on Test Dataset")
ax1.grid(True,alpha=0.2)
ax1.set_xticklabels([])
for x, y, label in zip(df.location, df.loss, df.location):
    ax1.text(x, y,          # 标注位置
             str(label),    # 显示的文字（df.a 字符串）
             ha='center',   # 水平居中
             va='bottom',   # 垂直在点下方（改成 top 就是在点上方）
             fontsize=6,    # 字体大小
             color='black')   # 颜色


# 绘制Accuracy
ax2.plot(df["location"], df["acc"], marker='o', color='green')
ax2.set_xlabel("Scenario")
ax2.set_xticklabels([])
ax2.set_ylabel("acc")
ax2.set_title("acc on Test Dataset")
ax2.grid(True,alpha=0.2)
for x, y, label in zip(df.location, df.acc, df.location):
    ax2.text(x, y,          # 标注位置
             str(label),    # 显示的文字（df.a 字符串）
             ha='center',   # 水平居中
             va='bottom',   # 垂直在点下方（改成 top 就是在点上方）
             fontsize=6,    # 字体大小
             color='black')   # 颜色

# 绘制Auc
ax3.plot(df["location"], df["auc"], marker='o', color='blue')
ax3.set_xlabel("Scenario")
ax3.set_xticklabels([])
ax3.set_ylabel("auc")
ax3.set_title("auc on Test Dataset")
ax3.grid(True,alpha=0.2)
for x, y, label in zip(df.location, df.auc, df.location):
    ax3.text(x, y,          # 标注位置
             str(label),    # 显示的文字（df.a 字符串）
             ha='center',   # 水平居中
             va='bottom',   # 垂直在点下方（改成 top 就是在点上方）
             fontsize=6,    # 字体大小
             color='black')   # 颜色

# 保存图片
plt.tight_layout()
plt.savefig(f"./results/metrics_On_Test.png",dpi=300)
plt.show()
plt.close()
print("\n Ada tell you ====== metrics On Test Dataset saved in metrics_On_Test.png")