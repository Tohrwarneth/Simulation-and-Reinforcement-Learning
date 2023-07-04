import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file, figure_name='Running average of previous 100 scores'):
    if isinstance(scores, list):
        running_avg = np.array(scores)
    else:
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title(figure_name)
    if figure_file is None:
        figure_file = 'images/'
        figure_file += figure_name.replace(' ', '_')
    plt.savefig(figure_file)
