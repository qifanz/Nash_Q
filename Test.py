from TwoPlayerEnv import *
from Player import *
from Plotter import *
import pickle


N_ITERATIONS = 5000

env = TwoPlayerEnv()
player = Player(env.get_n_states(),
                env.get_n_actions())
opponent = Player(env.get_n_states(),
                  env.get_n_actions())

reward_history = []

for iteration in range(N_ITERATIONS):
    cumulative_reward = env.play(player, opponent)
    reward_history.append(cumulative_reward)

    if iteration % 1000 == 0:
        print('Runned ',iteration, ' iterations')
        PLAYER_PKL = 'data/player_' + str(iteration) + '.pkl'
        OPPONENT_PKL = 'data/opponent_' + str(iteration) + '.pkl'
        REWARDS_PKL = 'data/rewards_' + str(iteration) + '.pkl'
        f = open(PLAYER_PKL, 'wb')
        pickle.dump(player, f)
        f.close()

        f = open(OPPONENT_PKL, 'wb')
        pickle.dump(opponent, f)
        f.close()

        f = open(REWARDS_PKL, 'wb')
        pickle.dump(reward_history, f)
        f.close()

plot_cumulative_rewards(reward_history)


