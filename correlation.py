import pandas as pd
import matplotlib.pyplot as plt

# 从 CSV 文件中读取数据
data = pd.read_csv('HW1.csv')

# 计算相关系数
correlation = data.corr()['song_popularity'].drop('song_popularity')

# 绘制条形图
plt.figure(figsize=(10, 6))
correlation.plot(kind='bar')
plt.xlabel('Input Features')
plt.ylabel('Correlation Coefficient')
plt.title('Correlation between Input Features and Song Popularity')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig('correlation.png')