import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('HW1.csv')


correlation = data.corr()['song_popularity'].drop('song_popularity')


plt.figure(figsize=(10, 6))
correlation.plot(kind='bar')
plt.xlabel('Input Features')
plt.ylabel('Correlation Coefficient')
plt.title('Correlation between Input Features and Song Popularity')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig('correlation.png')