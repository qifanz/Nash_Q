import matplotlib.pyplot as plt
import numpy as np


def plot_cumulative_rewards(cumulative_rewards):
    smooth_factor = 50
    #for i in range(0, len(cumulative_rewards), smooth_factor):
    #    cumulative_rewards[i:i + smooth_factor] = np.mean(cumulative_rewards[i:i + smooth_factor])

    plt.figure()
    title = 'Cumulative Rewards'
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.plot(cumulative_rewards)
    plt.show()
