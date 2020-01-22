import numpy as np
class Player:
    def __init__(self, n_states, n_actions):
        self.Q_table = np.zeros((n_states, n_actions, n_actions))

    def choose_action(self, state):
        return None

    def update_Q(self, reward, state, new_state):
        pass