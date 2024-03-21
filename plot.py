import pandas as pd
import matplotlib.pyplot as plt

# 从 CSV 文件中读取数据
data = pd.read_csv('prediction_train.csv', header=None)
# 设定列名
data.columns = ['Prediction', 'Actual', 'x3']
data.sort_values(by='x3', inplace=True)
# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(data['x3'][:], data['Prediction'][:], label='Prediction', s=3)
plt.scatter(data['x3'][:], data['Actual'][:], label='Actual', s=3)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison between Prediction and Actual')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('prediction.png')

plt.figure(figsize=(10, 6))
M=200
plt.plot(data['x3'][:M], data['Prediction'][:M], linestyle='-', color='blue', label='Prediction Line', linewidth=0.5)
plt.scatter(data['x3'][:M], data['Actual'][:M], linestyle='-', color='orange', label='Actual data', s=1)
plt.xlabel('Danceability')
plt.ylabel('Song_popularity')
plt.title('Comparison between Prediction and Actual')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('prediction_line.png')