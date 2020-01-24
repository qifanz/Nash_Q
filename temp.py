from Plotter import *
import pickle

f = open('./data/rewards_3000.pkl', 'rb')
rewards = pickle.load(f)
f.close()
plot_cumulative_rewards(rewards)