import pandas as pd
import matplotlib.pyplot as plt


def plot_fig(file_name):
    data = pd.read_csv(file_name, header=None)
    data.columns = ['Prediction', 'Actual', 'x3']
    data.sort_values(by='x3', inplace=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(data['x3'][:], data['Prediction'][:], label='Prediction', s=3)
    plt.scatter(data['x3'][:], data['Actual'][:], label='Actual', s=3)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Comparison between Prediction and Actual')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(file_name + ".png")

    plt.figure(figsize=(10, 6))
    M=200
    plt.plot(data['x3'][:], data['Prediction'][:], linestyle='-', color='blue', label='Prediction Line', linewidth=0.5)
    plt.scatter(data['x3'][:], data['Actual'][:], linestyle='-', color='orange', label='Actual data', s=1)
    plt.xlabel('Danceability')
    plt.ylabel('Song_popularity')
    plt.title('Comparison between Prediction and Actual')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(file_name + "_curve.png")



plot_fig("prediction_train.csv")
plot_fig("prediction_test.csv")
plot_fig("fivefold_test_1.csv")