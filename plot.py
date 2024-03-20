import pandas as pd
import matplotlib.pyplot as plt

# 从 CSV 文件中读取数据
data = pd.read_csv('prediction.csv', header=None)
# 设定列名
data.columns = ['Prediction', 'Actual']

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(range(1, 51), data['Prediction'][:50], label='Prediction', s=3)
plt.scatter(range(1, 51), data['Actual'][:50], label='Actual', s=3)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison between Prediction and Actual')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('prediction.png')

plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), data['Prediction'][:50], linestyle='-', color='blue', label='Prediction Line')
plt.plot(range(1, 51), data['Actual'][:50], linestyle='-', color='orange', label='Actual Line')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison between Prediction and Actual')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('prediction_line.png')