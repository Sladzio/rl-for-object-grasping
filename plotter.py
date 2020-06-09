import numpy as np
from stable_baselines.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_rewards(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """

    fig = plt.figure(title, figsize=(8, 4))
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)

    for (label, log) in log_folder:
        x, y = ts2xy(load_results(log), 'timesteps')
        y = moving_average(y, window=100)
        # Truncate x
        x = x[len(x) - len(y):]
        plt.plot(x, y, label=label)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(title+'.png')

