import numpy as np
from MatrixGameSolver import *


class Player:
    def __init__(self, n_states, n_actions, discount_factor=0.9, learning_rate=0.05):
        self.beta = discount_factor
        self.lr = learning_rate
        self.n_actions = n_actions
        self.Q_table = np.zeros((n_states, n_actions, n_actions))

    def choose_action(self, state):
        equilibrium, policy = self.solve_nash_at_state(state)
        action = self.random_choose_action_from_policy(self.normalize_policy(policy))
        return action

    def normalize_policy(self, policy):
        return np.divide(policy, np.sum(policy))

    def random_choose_action_from_policy(self, policy):
        return np.random.choice(np.arange(0, self.n_actions), p=policy)

    def observe(self, reward, state, action_self, action_opponent, new_state):
        new_equilibrium, _ = self.solve_nash_at_state(new_state)
        self.Q_table[state, action_self, action_opponent] \
            = (1 - self.lr) * self.Q_table[state, action_self, action_opponent] \
              + self.lr * (reward + self.beta * new_equilibrium)

    def solve_nash_at_state(self, state):
        M = self.Q_table[state]
        equilibrium, policy, policy_opponent = linprog_solve(M)
        return equilibrium, policy
