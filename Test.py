from TwoPlayerEnv import *
from Player import *

N_ITERATIONS = 10000

env = TwoPlayerEnv()
player = Player(env.get_n_states(),
                env.get_n_actions())
opponent = Player(env.get_n_states(),
                  env.get_n_actions())

for iteration in range(N_ITERATIONS):
    env.play(player, opponent)
